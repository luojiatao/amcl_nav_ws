#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/LaserScan.h>
#include <champion_nav_msgs/ChampionNavLaserScan.h>
#include <iostream>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sensor_msgs/PointCloud2.h> 
#include <pcl_conversions/pcl_conversions.h> 
//如果使用调试模式，可视化点云，需要安装PCL
#define debug_ 0

//#if debug_
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>
pcl::visualization::CloudViewer g_PointCloudView("PointCloud View");//初始化一个pcl窗口
//#endif

class LidarMotionCalibrator
{
public:
    //构造函数，初始化tf_、订阅者、回调函数ScanCallBack
    LidarMotionCalibrator(tf::TransformListener* tf)
    {
        tf_ = tf;
        scan_sub_ = nh_.subscribe(scan_sub_name_, 10, &LidarMotionCalibrator::ScanCallBack, this);
        scan_pub_ = nh_.advertise<champion_nav_msgs::ChampionNavLaserScan>(scan_pub_name_, 1000);
        pcl_pub = nh_.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1000);   
    }
    //析构函数，释放tf_
    ~LidarMotionCalibrator()
    {
        if(tf_!=NULL)
            delete tf_;
    }
    // 拿到原始的激光数据来进行处理
    void ScanCallBack(const sensor_msgs::LaserScanConstPtr& scan_msg)
    {
        std::cout<<"get scan_raw topic"<<std::endl;
        //转换到矫正需要的数据
        ros::Time startTime, endTime;
        //一帧scan数据到来首先得出，开始结束的时间戳、数据的size
        startTime = scan_msg->header.stamp;
        sensor_msgs::LaserScan laserScanMsg = *scan_msg;

        //得到最终点的时间
        int beamNum = laserScanMsg.ranges.size();
        endTime = startTime + ros::Duration(laserScanMsg.time_increment * beamNum);

        // 将数据复制出来
        std::vector<double> angles,ranges;
        for(int i = 0; i < beamNum;i++)
        {
            double lidar_dist = laserScanMsg.ranges[i];//单位米
            double lidar_angle = laserScanMsg.angle_min + laserScanMsg.angle_increment * i;//单位弧度

            ranges.push_back(lidar_dist);
            angles.push_back(lidar_angle);
        }
        //#if debug_
        visual_cloud_.clear();
        //转换为pcl::pointcloud for visuailization
        //数据矫正前、封装打算点云可视化、红色
        visual_cloud_scan(ranges,angles,255,0,0,startTime);
        //#endif
        std::cout<<"start Lidar_Calibration"<<std::endl;
        //进行矫正
        Lidar_Calibration(ranges,angles,
                          startTime,
                          endTime,
                          tf_);
        //数据矫正后、封装打算点云可视化、绿色
        //转换为pcl::pointcloud for visuailization
        //#if debug_
        visual_cloud_scan(ranges,angles,0,255,0,startTime);
        //#endif
        //发布矫正后的scan
        //ROS_INFO("scan_time:%f",ros::Duration(laserScanMsg.time_increment * beamNum).toSec());
        std::cout<<"start pub_scan topic"<<std::endl;
        scan_cal_pub(ranges,angles,startTime,ros::Duration(laserScanMsg.time_increment * beamNum).toSec());
        pub_pcl_scan(ranges,angles,startTime);
        //进行显示
        //#if debug_
        g_PointCloudView.showCloud(visual_cloud_.makeShared());
        //#endif
    }

    //矫正之后的scan数据的发布函数，这里默认发布所有360度的所有数据,正着安装,所填写的参数是rplidar A1
    void scan_cal_pub(const std::vector<double> &ranges,std::vector<double> &angles,ros::Time pubTime,double scan_time)
    {
        //定义sensor_msgs::LaserScan数据
        champion_nav_msgs::ChampionNavLaserScan tempScan;
        tempScan.header.stamp=pubTime;
        tempScan.header.frame_id=scan_frame_name_;
        tempScan.angle_min=-3.12413907051;
        tempScan.angle_max=3.14159274101;
        //tempScan.angle_increment=(tempScan.angle_max-tempScan.angle_min)/(ranges.size()-1);//这个要注意
        tempScan.scan_time=(float)scan_time;                                                //单位s
        tempScan.time_increment=scan_time/(ranges.size()-1);                               //这个是变化的
        
        tempScan.range_min=0.15;
        tempScan.range_max=6.0;
        
        tempScan.ranges.resize(ranges.size());
        tempScan.angles.resize(angles.size());
        tempScan.intensities.resize(ranges.size());

        //填充雷达数据,判断填充的是否正确
        for(size_t i=0;i<ranges.size();++i)
        {
            tempScan.ranges[i]=ranges[i];
             tempScan.angles[i]=angles[i];
            tempScan.intensities[i]=15.0;//这个可能没用
        }

        //===========================================================================
        #if debug_
        //封装数据的可视化的测试
        std::vector<double> angles_temp,ranges_temp;
        for(int i = 0; i < tempScan.ranges.size();i++)
        {
            double lidar_dist = tempScan.ranges[i];
            double lidar_angle = tempScan.angle_min + tempScan.angle_increment * i;

            ranges_temp.push_back(lidar_dist);
            angles_temp.push_back(lidar_angle);
        }
        visual_cloud_scan(ranges_temp,angles_temp,255,255,255);
        #endif
        //===========================================================================
        //发布
        scan_pub_.publish(tempScan);
    }


    //pub scan as pcl
    void pub_pcl_scan(const std::vector<double> &ranges_,const std::vector<double> &angles_,ros::Time startTime)
    {
        tf::Stamped<tf::Pose> visualPose;
        if (!getLaserPose(visualPose, startTime, tf_)) {
            ROS_WARN("Not visualPose,Can not Calib");
            return;
        }
 
        double visualYaw = tf::getYaw(visualPose.getRotation());

        pcl::PointCloud<pcl::PointXYZ> cloud; 
	    sensor_msgs::PointCloud2 output; 
        cloud.width = ranges_.size(); 
	    cloud.height = 1; //此处也可以为cloud.width = 4; cloud.height = 2; 
        cloud.points.resize(cloud.width * cloud.height); 
        for(int i = 0; i < ranges_.size();i++)
        {
            if(ranges_[i] < 0.05 || std::isnan(ranges_[i]) || std::isinf(ranges_[i]))
                continue;
            
            // cloud.points[i].x = ranges_[i] * cos(angles_[i]);
            // cloud.points[i].y = ranges_[i] * sin(angles_[i]);
            // cloud.points[i].z = 1.0;
            double x = ranges_[i] * cos(angles_[i]);
            double y = ranges_[i] * sin(angles_[i]);
 
            pcl::PointXYZRGB pt;
            cloud.points[i].x = x * cos(visualYaw) - y * sin(visualYaw) + visualPose.getOrigin().getX();
            cloud.points[i].y = x * sin(visualYaw) + y * cos(visualYaw) + visualPose.getOrigin().getY();
            cloud.points[i].z = 0;
        }        
        //Convert the cloud to ROS message 
        pcl::toROSMsg(cloud, output); 
        output.header.frame_id = odom_name_;
        pcl_pub.publish(output); 
    }

    //使用点云将激光可视化
    //#if debug_
    void visual_cloud_scan(const std::vector<double> &ranges_,const std::vector<double> &angles_,unsigned char r_,unsigned char g_,unsigned char b_,ros::Time startTime)
    {
        tf::Stamped<tf::Pose> visualPose;
        if (!getLaserPose(visualPose, startTime, tf_)) {
            ROS_WARN("Not visualPose,Can not Calib");
            return;
        }
 
        double visualYaw = tf::getYaw(visualPose.getRotation());

        unsigned char r = r_, g = g_, b = b_; //变量不要重名 
        for(int i = 0; i < ranges_.size();i++)
        {
            if(ranges_[i] < 0.05 || std::isnan(ranges_[i]) || std::isinf(ranges_[i]))
                continue;
            
            pcl::PointXYZRGB pt;
            pt.x = ranges_[i] * cos(angles_[i]);
            pt.y = ranges_[i] * sin(angles_[i]);
            pt.z = 1.0;
            
           /*
            double x = ranges_[i] * cos(angles_[i]);
            double y = ranges_[i] * sin(angles_[i]);
 
            pcl::PointXYZRGB pt;
            pt.x = x * cos(visualYaw) - y * sin(visualYaw) + visualPose.getOrigin().getX();
            pt.y = x * sin(visualYaw) + y * cos(visualYaw) + visualPose.getOrigin().getY();
            pt.z = 1.0;
        */

            // pack r/g/b into rgb
            unsigned int rgb = ((unsigned int)r << 16 | (unsigned int)g << 8 | (unsigned int)b);
            pt.rgb = *reinterpret_cast<float*>(&rgb);

            visual_cloud_.push_back(pt);
        }        
    }
    //#endif 
    /**
     * @name getLaserPose()
     * @brief 得到机器人在里程计坐标系中的位姿tf::Pose
     *        得到dt时刻激光雷达在odom坐标系的位姿odom_pose
     * @param odom_pos  机器人的位姿
     * @param dt        dt时刻
     * @param tf_
    */
    bool getLaserPose(tf::Stamped<tf::Pose> &odom_pose,
                      ros::Time dt,
                      tf::TransformListener * tf_)
    {
        odom_pose.setIdentity();

        tf::Stamped < tf::Pose > robot_pose;
        robot_pose.setIdentity();
        robot_pose.frame_id_ = scan_frame_name_;//这里是laser_link
        robot_pose.stamp_ = dt;                 //设置为ros::Time()表示返回最近的转换关系

        // get the global pose of the robot
        try
        {   //解决时间不同步问题
            if(!tf_->waitForTransform(odom_name_, scan_frame_name_, dt, ros::Duration(0.5)))             // 0.15s 的时间可以修改
            {
                ROS_ERROR("LidarMotion-Can not Wait Transform()");
                return false;
            }
            tf_->transformPose(odom_name_, robot_pose, odom_pose);
        }
        catch (tf::LookupException& ex)
        {
            ROS_ERROR("LidarMotion: No Transform available Error looking up robot pose: %s\n", ex.what());
            return false;
        }
        catch (tf::ConnectivityException& ex)
        {
            ROS_ERROR("LidarMotion: Connectivity Error looking up looking up robot pose: %s\n", ex.what());
            return false;
        }
        catch (tf::ExtrapolationException& ex)
        {
            ROS_ERROR("LidarMotion: Extrapolation Error looking up looking up robot pose: %s\n", ex.what());
            return false;
        }

        return true;
    }

    /**
     * @brief Lidar_MotionCalibration
     *        在分段时刻的，激光雷达运动畸变去除;
     *        在此分段函数中，认为机器人是匀速运动；
     * @param frame_base_pose       标定完毕之后的基准坐标系
     * @param frame_start_pose      本分段第一个激光点对应的位姿
     * @param frame_end_pose        本分段最后一个激光点对应的位姿
     * @param ranges                激光数据－－距离
     * @param angles                激光数据－－角度
     * @param startIndex            本分段第一个激光点在激光帧中的下标
     * @param beam_number           本分段的激光点数量
     */

    void Lidar_MotionCalibration(
    tf::Stamped<tf::Pose> frame_base_pose,
    tf::Stamped<tf::Pose> frame_start_pose,
    tf::Stamped<tf::Pose> frame_end_pose,
    std::vector<double>& ranges,
    std::vector<double>& angles,
    int startIndex,
    int& beam_number)
    {
        //TODO
        // frame_base_pose.inverse;
        tf::Quaternion start_q = frame_start_pose.getRotation();
        tf::Quaternion end_q = frame_end_pose.getRotation();
        tf::Vector3 start_xy(frame_start_pose.getOrigin().getX(), frame_start_pose.getOrigin().getY(), 1);
        tf::Vector3 end_xy(frame_end_pose.getOrigin().getX(), frame_end_pose.getOrigin().getY(), 1);
 
        for (size_t i = startIndex; i < startIndex + beam_number; i++) {
            tf::Vector3 mid_xy = start_xy.lerp(end_xy, (i - startIndex) / (beam_number - 1));
            tf::Quaternion mid_q = start_q.slerp(end_q, (i - startIndex) / (beam_number - 1));
            tf::Transform mid_frame(mid_q, mid_xy);
            double x = ranges[i] * cos(angles[i]);
            double y = ranges[i] * sin(angles[i]);
            tf::Vector3 calib_point = frame_base_pose.inverse() * mid_frame * tf::Vector3(x, y, 1);
            ranges[i] = sqrt(calib_point[0] * calib_point[0] + calib_point[1] * calib_point[1]);
            angles[i] = atan2(calib_point[1], calib_point[0]);
        }
        //end of TODO
    }

    /*
    void Lidar_MotionCalibration(
            tf::Stamped<tf::Pose> frame_base_pose,//对于每一帧scan，基准坐标系一致
            tf::Stamped<tf::Pose> frame_start_pose,
            tf::Stamped<tf::Pose> frame_end_pose,
            std::vector<double>& ranges,
            std::vector<double>& angles,
            int startIndex,  //每一个分段的激光点起始序号
            int& beam_number)//此分段中，激光点的个数
    {
        //每个位姿进行线性插值时的步长
        double beam_step = 1.0 / (beam_number-1);

        //机器人的起始角度 和 最终角度，四元数表示
        tf::Quaternion start_angle_q =   frame_start_pose.getRotation();
        tf::Quaternion   end_angle_q =   frame_end_pose.getRotation();

        //转换到弧度
        double start_angle_r = tf::getYaw(start_angle_q);
        double base_angle_r = tf::getYaw(frame_base_pose.getRotation());

        //机器人的起始位姿
        tf::Vector3 start_pos = frame_start_pose.getOrigin();//Ps
        start_pos.setZ(0);

        //最终位姿
        tf::Vector3 end_pos = frame_end_pose.getOrigin();    //Pe
        end_pos.setZ(0);

        //基础坐标系
        tf::Vector3 base_pos = frame_base_pose.getOrigin();
        base_pos.setZ(0);

        double mid_angle;
        tf::Vector3 mid_pos;
        tf::Vector3 mid_point;

        double lidar_angle, lidar_dist;
        //插值计算出来每个点对应的位姿
        for(int i = 0; i< beam_number;i++)
        {
            //得到该激光点的角度插值，线性插值需要步长、起始和结束数据
            mid_angle =  tf::getYaw(start_angle_q.slerp(end_angle_q, beam_step * i));

            //得到该激光点的里程计位姿线性插值
            mid_pos = start_pos.lerp(end_pos, beam_step * i);

            //得到激光点在odom 坐标系中的坐标 根据
            double tmp_angle;

            //如果激光雷达不等于无穷,则需要进行矫正.//首先读数据进行判断
            if( tfFuzzyZero(ranges[startIndex + i]) == false)
            {
                //计算对应的激光点在odom坐标系中的坐标

                //得到这帧激光束距离和夹角
                lidar_dist  =  ranges[startIndex+i];
                lidar_angle =  angles[startIndex+i];

                //在激光雷达坐标系下激光点的坐标
                double laser_x,laser_y;
                laser_x = lidar_dist * cos(lidar_angle);
                laser_y = lidar_dist * sin(lidar_angle);

                //在对应的里程计坐标系下激光点的坐标
                double odom_x,odom_y;
                odom_x = laser_x * cos(mid_angle) - laser_y * sin(mid_angle) + mid_pos.x();
                odom_y = laser_x * sin(mid_angle) + laser_y * cos(mid_angle) + mid_pos.y();

                //转换到类型中去
                mid_point.setValue(odom_x, odom_y, 0);
                //得到在基准坐标系下激光点的坐标
                //把在odom坐标系中的激光数据点 转换到 基础坐标系
                //得到那一瞬时，应该测得的激光点的数据
                double x0,y0,a0,s,c;
                x0 = base_pos.x();
                y0 = base_pos.y();
                a0 = base_angle_r;
                s = sin(a0);
                c = cos(a0);
                // 
                //  * 把base转换到odom 为[c -s x0;
                //  *                     s  c y0;
                //  *                     0  0 1 ]
                //  * 把odom转换到base为 [c s -x0*c - y0*s;
                //  *                    -s c  x0*s - y0*c;
                //  *                     0 0  1          ]
                //  
                double tmp_x,tmp_y;
                tmp_x =  mid_point.x()*c  + mid_point.y()*s - x0*c - y0*s;
                tmp_y = -mid_point.x()*s  + mid_point.y()*c  + x0*s - y0*c;
                mid_point.setValue(tmp_x,tmp_y,0);

                //然后计算该激光点以起始坐标为起点的 dist angle
                double dx,dy;
                dx = (mid_point.x());
                dy = (mid_point.y());
                lidar_dist = sqrt(dx*dx + dy*dy);
                lidar_angle = atan2(dy,dx);

                //激光雷达被矫正
                ranges[startIndex+i] = lidar_dist;
                angles[startIndex+i] = lidar_angle;
            }
            //如果等于无穷,则随便计算一下角度
            else
            {
                //激光角度
                lidar_angle = angles[startIndex+i];

                //里程计坐标系的角度
                tmp_angle = mid_angle + lidar_angle;
                tmp_angle = tfNormalizeAngle(tmp_angle);

                //如果数据非法 则只需要设置角度就可以了。把角度换算成start_pos坐标系内的角度
                lidar_angle = tfNormalizeAngle(tmp_angle - start_angle_r);

                angles[startIndex+i] = lidar_angle;
            }
        }
    }
    */

    //激光雷达数据　分段线性进行插值　分段的周期为5ms，可以更改
    //这里会调用Lidar_MotionCalibration()
    /**
     * @name Lidar_Calibration()
     * @brief 激光雷达数据　分段线性进行差值　分段的周期为5ms
     * @param ranges 激光束的距离值集合
     * @param angle　激光束的角度值集合
     * @param startTime　第一束激光的时间戳
     * @param endTime　最后一束激光的时间戳
     * @param *tf_
    */
    void Lidar_Calibration(std::vector<double>& ranges,
                           std::vector<double>& angles,
                           ros::Time startTime,
                           ros::Time endTime,
                           tf::TransformListener * tf_)
    {
        //统计激光束的数量
        int beamNumber = ranges.size();
        if(beamNumber != angles.size())
        {
            ROS_ERROR("Error:ranges not match to the angles");
            return ;
        }

        // 5000us来进行分段
        int interpolation_time_duration = 5 * 1000;//单位us

        tf::Stamped<tf::Pose> frame_base_pose;
        tf::Stamped<tf::Pose> frame_start_pose;
        tf::Stamped<tf::Pose> frame_mid_pose;
        tf::Stamped<tf::Pose> frame_end_pose;

        //起始时间 us
        double start_time = startTime.toSec() * 1000 * 1000;    //转化单位为us
        double end_time = endTime.toSec() * 1000 * 1000;
        double time_inc = (end_time - start_time) / beamNumber; // 每束激光数据的时间间隔，单位us

        //当前插值的段的起始坐标
        int start_index = 0;

        //起始点的位姿 这里要得到起始点位置的原因是　起始点就是我们的base_pose
        //所有的激光点的基准位姿都会改成我们的base_pose
        
        //得到t时刻激光雷达在odom坐标系的位姿frame_start_pose、frame_end_pose
        if(!getLaserPose(frame_start_pose, ros::Time(start_time /1000000.0), tf_))
        {
            ROS_WARN("Not Start Pose,Can not Calib");
            return ;
        }

        if(!getLaserPose(frame_end_pose,ros::Time(end_time / 1000000.0),tf_))
        {
            ROS_WARN("Not End Pose, Can not Calib");
            return ;
        }
        //计数报错使用
        int cnt = 0;
        //基准坐标就是第一个位姿的坐标
        frame_base_pose = frame_start_pose;
        for(int i = 0; i < beamNumber; i++)
        {
            //分段线性,时间段的大小为interpolation_time_duration=5000us
            double mid_time = start_time + time_inc * (i - start_index);//这里的mid_time、start_time多次重复利用
            if(mid_time - start_time > interpolation_time_duration || (i == beamNumber - 1))
            {
                cnt++;
                //得到临时终点frame_mid_pose在里程计中的位姿，对应一个激光束
                if(!getLaserPose(frame_mid_pose, ros::Time(mid_time/1000000.0), tf_))
                {
                    ROS_ERROR("Mid %d Pose Error",cnt);
                    return ;
                }
                //对当前的起点和终点进行插值
                //interpolation_time_duration分段间隔中间有多少个点，算上本分段间隔首尾
                int interp_count = i - start_index + 1;//可以尝试推算一下，通常会有几十个或者上百个激光点
                //对本分段的激光点进行运动畸变的去除
                Lidar_MotionCalibration(frame_base_pose,
                                        frame_start_pose,
                                        frame_mid_pose,
                                        ranges,
                                        angles,
                                        start_index,
                                        interp_count);

                //更新时间
                start_time = mid_time;
                start_index = i;     //为了方便计算分段中激光点个数
                frame_start_pose = frame_mid_pose;
            }
        }
    }

public:
    //声明TF的聆听者、ROS句柄、scan的订阅者、scan的发布者
    tf::TransformListener* tf_;
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher  scan_pub_;
    ros::Publisher  pcl_pub;

    //针对各自的情况需要更改的名字，自行更改
    const std::string scan_frame_name_="laser_frame";
    const std::string odom_name_="odom";
    const std::string scan_sub_name_="scan_raw";
    const std::string scan_pub_name_="scan";

    //#if debug_
    //可视化点云对象
    pcl::PointCloud<pcl::PointXYZRGB> visual_cloud_;
    //#endif
};

int main(int argc,char ** argv)
{
    ros::init(argc,argv,"LidarMotionCalib");

    tf::TransformListener tf(ros::Duration(10.0));

    LidarMotionCalibrator tmpLidarMotionCalib(&tf);

    ros::spin();
    return 0;
}

