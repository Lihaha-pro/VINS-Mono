#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}
/**
 * @brief 外参，重投影置信度（1.5个像素），延时设置
 * 
 */
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);//平移外参设置给特征点管理器
    //使用了虚拟相机，重投影精度（提取特征点精度）设置为1.5个像素
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}
/**
 * @brief 处理Image IMU数据组中一个imu数据时调用：对imu数据进行处理，包括更新预积分量，提供优化状态量的初值
 * 
 * @param dt 
 * @param linear_acceleration 
 * @param angular_velocity 
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    //滑窗中保留11帧，frame_count表示现在正在处理第几帧，最大就到11了（另外，实际上11帧形成了10个帧间约束，这也是为什么第一帧图像前的imu数据没有用）
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);//!通过该函数进入预积分部分!!!!!!!!!!!!!!!!!非常关键！
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);//初始化使用
        //保存传感器数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        //!利用IMU数据，作为优化初值
        //有一个中值积分，更新滑窗中状态量，本质是给非线性优化提供可信的初始值（计算方法和之前pub出去IMU数据一样）
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}
/**
 * @brief addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧               
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param header 图片数据header，主要目的是获得时间戳
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    //Step 1 将特征点信息加到f_manager这个特征点管理器中，同时进行是否是关键帧的检查
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        //如果上一帧是关键帧，则去掉滑窗中最老的帧
        marginalization_flag = MARGIN_OLD;
    else
        //如果上一帧不是关键帧，就移除滑窗中的上一帧
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    //all_image_frame用来做初始化相关操作，他保留滑窗起始到当前的所有帧
    //初始化时不会margin掉任何一帧，即使被标记
    //这里是将光流结果和预积分结果送入imageframe管理
    ImageFrame imageframe(image, header.stamp.toSec());//图像帧对象
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));//包含了所有的imageframe，用来实现初始化
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};//重新初始化预积分量

    //Step 2 外参初始化
    if(ESTIMATE_EXTRINSIC == 2)//没有任何先验信息
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);///获得当前帧和上一帧的匹配特征点坐标
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))///进行外参旋转矩阵的标定
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)//有足够帧数
        {
            bool result = false;
            //要有可信的外参值，同时距离上次初始化不成功至少0.1s
            //Step 3 VIO初始化
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();///视觉惯性联合初始化
               initial_timestamp = header.stamp.toSec();//更新初始化时间戳
            }
            if(result)
            {
                solver_flag = NON_LINEAR;//标志位改成非线性优化（之前一直是初始化状态）
                //Step 4 非线性优化求解VIO
                solveOdometry();
                //Step 5 滑动窗口
                slideWindow();
                //Step 6 移除无效地图点
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();//1
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())//2
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();//3
        f_manager.removeFailures();//4取出估计失败的点并发布特征点位置
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();//5
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];//6
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}
/**
 * @brief VIO初始化，将滑窗中的PVQ恢复到第0帧并且和重力对齐
 * Step 1 检查是否有足够的视差激励
 * Step 2 单目纯视觉slam：1.选定枢纽帧，计算相对位姿 2.计算各KF的位姿，并恢复3D点坐标，进而使用ceres优化
 * Step 3 初始化恢复其他非KF的帧的位姿
 * Step 4 视觉惯性对齐
 * 
 * @return true 
 * @return false 
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //Step 1 check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        //从第二帧开始检查imu
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            //累加每一帧带重力加速度的deltav
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            //求方差
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));//计算标准差
        //ROS_WARN("IMU variation %f!", var);
        //这里实际上并没有用到，作者注释掉了
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    //Step 2 global sfm 纯视觉单目slam
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;//保存每个特征点的信息
    //遍历所有特征点
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;//改变量和imu没有关系，就是存储观测特征点的帧的索引
        SFMFeature tmp_feature;//用来后续做sfm
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        //遍历当前特征点的观测信息
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;//枢纽帧在滑窗中的索引
    if (!relativePose(relative_R, relative_T, l))///使用对极几何得到枢纽帧到尾帧的位姿变换
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    ///使用ceres优化得到各帧位姿及地图点坐标
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }
    //Step 3 初始化阶段，处理所有的非KF帧进行位姿求解
    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        //如果这一帧本身就是KF，直接得到位姿
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        //更新寻找关键帧的索引
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        //最近的KF提供一个初始值，Twc->Tcw
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        //遍历该帧的特征点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            //单目，所以就执行一次
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())//如果找到三角化的3d点
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        //使用opencv的pnp接口
        /** 
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)   
         */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        //cv->eigen Tcw->Twc
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        //Twc->Twi
        //由于尺度未恢复，所以位移暂时不移动到imu系
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    //Step 4 视觉惯性对齐（视觉slam没有尺度）
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}
/**
 * @brief 视觉惯性对齐入口函数
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate  
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return true 
 * @return false 
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    //首先将对齐后的KF的位姿赋值给滑窗中的值Rwi twc
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    //所有特征点深度设为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    //重新计算特征点深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    //多约束三角化所有的特征点，注意仍然是尺度模糊的
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    //将滑窗中的预积分重新计算
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    //下面吧所有的状态对齐到第0帧imu的坐标系
    for (int i = frame_count; i >= 0; i--)
        //Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    //把求解出来的KF速度赋值给滑窗
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            //当时求得速度为imu系，现在转到world系
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    //把尺度模糊的3D点恢复到真实尺度下
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }
    //所有的PVQ全部对齐到第0帧，同时和对齐的重力方向，重力方向转到z轴上
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    //所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];//全部对齐到重力方向下
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}
/**
 * @brief 选定滑窗中的枢纽帧，计算位姿变换（使用E矩阵）
 * 
 * @param relative_R 计算得到的旋转矩阵
 * @param relative_T 
 * @param l 枢纽帧在滑窗中的索引
 * @return true 
 * @return false 
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    //优先从最前边的帧开始，也就是id=0
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        //匹配点对大于20个
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();//计算视差（使用了向量范数）
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());//平均视差
            //虚拟相机下，平均视差要大于30个像素
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}
/**
 * @brief //三角化求解所有特征点的深度，并进行非线性优化
 * 
 */
void Estimator::solveOdometry()
{
    //保证滑窗中帧满了
    if (frame_count < WINDOW_SIZE)
        return;
    //齐次要求初始化已经完成
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        //先把应该三角化但没有三角化的特征点三角化，方便后边提供约束
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}
/**
 * @brief 优化过程开始前将待优化变量存入double数组
 * 
 */
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }
    //特征点逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    //传感器时间同步
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}
/**
 * @brief 数据转换，vector2double的相反过程
 * 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
 * 也就是恢复到原始第一帧的位姿下
 */
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    //优化后的位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    //得到优化前后位姿的差yaw
    double y_diff = origin_R0.x() - origin_R00.x();

    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }
    //根据位姿差做修正，保证第一帧优化前后位姿不变
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        //保持第1帧yaw不变
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        //保持第1帧平移不变
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief 基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 *        添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，IMU的外参也可以加进来
 *        添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *        根据倒数第二帧是不是关键帧确定边缘化的结果    
 * 
 */
void Estimator::optimization()
{
    //ceres进行非线性优化
    ceres::Problem problem;
    ceres::LossFunction *loss_function;//定义核函数
    loss_function = new ceres::CauchyLoss(1.0);
    //loss_function = new ceres::HuberLoss(1.0);
    //Step 1 添加ceres参数块，类似g2o顶点
    //因为ceres用的是double数组，所以在下面用vector2double做类型装换
    //Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    //参数块 1：滑窗中图片位置，姿态，速度，零偏
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        //由于姿态不满足正常加法，即李群上不满足加法，因此要特殊定义其加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    //参数块 2：相机imu间外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            //如果不需要优化外参，就设置为fix
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    //参数块 3：传感器时间同步
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }
    //实际上还有地图点，其实平凡的参数块不需要调用AddParameterBlock，增加残差接口的时候会自动绑定
    TicToc t_whole, t_prepare;
    //eigen->double
    vector2double();
    //Step 2 通过残差约束来添加残差块，类似g2o的边
    //上一次边缘化结果作为本次的先验
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    ///imu预积分约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        //时间过长这个约束就不可信了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    ///视觉重投影约束
    //遍历每一个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;//必须满足出现两次并且在倒数第二帧之前出现过
 
        ++feature_index;//统计有效特征点数量

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;//第i帧归一化相机坐标

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {//遍历当前特征在每一帧中的信息
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;//得到第二个特征点坐标
            //带有时间延迟的一种形式，并没有走这个，但不小心//llh注释到这里了
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);//传入参数即同一3D点在两帧中的位姿
                //约束的变量是观测到该点的第一帧以及某一帧，以及外参和特征点深度
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());
    //回环检测的相关约束
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);//需要优化的回环帧位姿
        int retrive_feature_index = 0;
        int feature_index = -1;
        //遍历现有地图点
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;//起始帧id
            if(start <= relo_frame_local_index)//这个地图点能被对应的当前帧看到
            {
                //寻找回环帧能看到的地图点
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                //这个地图点也能被回环帧看到
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    //构造重投影约束，该地图点起始帧和回环帧之间
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }
    //Step 3 ceres优化求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;//这里可以用LM
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;//如果是边缘化老帧，操作较多，给的优化时间就短一些//?
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);//执行优化之后，优化的变量还在double数组中
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());
    //double->Eigen 优化结果保存到Eigen等变量中
    double2vector();
    //Step 4 边缘化
    TicToc t_whole_marginalization;
    //分为两种情况，边缘化最老的帧还是次新帧
    if (marginalization_flag == MARGIN_OLD)
    {
        //一个用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        //类似于手写高斯牛顿，将变量转化为double数组
        vector2double();
        //关于边缘化有几点注意：
        //1.找到需要边缘化的参数块，这里是第0帧位姿、速度、零偏，地图点
        //2.找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，即预积分约束，重投影约束，以及上一次边缘化约束
        //3.这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏
        
        //上一次边缘化结果
        // 先验误差会一直保存，而不是只使用一次
        // 如果上一次边缘化的信息存在
        // 要边缘化的参数块是 para_Pose[0] para_SpeedBias[0] 以及 para_Feature[feature_index](滑窗内的第feature_index个点的逆深度)
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {//查询last_marginalization_parameter_blocks中是首帧状态量的序号
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);//用drop_set把最老帧的先验信息干掉了
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        //只有第1个预积分和待边缘化参数块相连
        ///把要margin的IMU信息加进来
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                //和构建ceres问题一样，也要计算残差和雅克比
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);//这里的1是指第0帧（最老的帧）和第1帧的约束
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});//这里的0和1表示索引为0和1的参数块要被边缘化掉
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        //把要margin的视觉信息加进来
        {//添加视觉的先验，只添加起始帧是旧帧且观测次数大于2的Features
            int feature_index = -1;
            //遍历特征点
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();//该特征点被观测到的次数
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;//Feature的观测次数不小于2次，且起始帧不属于最后两帧

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                //保留起始自当前滑窗第0帧的特征点
                if (imu_i != 0)
                    continue;
                //得到该Feature在起始下的归一化坐标
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                //遍历观测到该特征点的其他KF，建立约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    //根据是否约束延时确定残差帧
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});//第0帧和3D点被margin
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        //Step 将三个ResidualBlockInfo中的参数块综合到marginalization_info中
        //其中，计算所有ResidualBlock(残差项)的残差和雅克比,parameter_block_data是参数块的容器。
        ///边缘化预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        ///边缘化操作
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}
/**
 * @brief 接收回环帧的消息
 * 
 * @param _frame_stamp 
 * @param _frame_index 
 * @param _match_points 
 * @param _relo_t 
 * @param _relo_r 
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    //在滑窗中寻找当前帧，因为VIO送个回环节点的是倒数第三帧，因此很有可能该帧还在滑窗内
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;//对应滑窗中的第i帧
            relocalization_info = 1;    //这是一个有效的回环信息
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];//借助VIO优化回环帧位姿，初值先设为当前帧位姿
        }
    }
}

