#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}
/**
 * @brief 标定imu和相机之间的旋转外参，通过imu和图像计算的旋转使用手眼标定计算获得
 * 
 * @param corres 
 * @param delta_q_imu 
 * @param calib_ric_result 输出参数，也就是计算得到的旋转外参
 * @return true 
 * @return false 
 */
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    //根据特征关联求解两个连续帧相机的旋转R12
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    //通过外参吧imu的旋转转移到相机坐标系，用于计算两种旋转的差值，从而给出一个核函数
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);//ric是上一次求解得到的外参

    Eigen::MatrixXd A(frame_count * 4, 4);//超定方程的系数矩阵
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        //使用Eigen接口计算相机和imu两个旋转矩阵的角度差值
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        //做了一个鲁棒核函数huber核
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;
        //把四元数转换为旋转矩阵
        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);//系数矩阵块加入鲁棒核函数（核函数作用在残差上）
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);//SVD分解
    Matrix<double, 4, 1> x = svd.matrixV().col(3);//取出V矩阵的最后一列
    Quaterniond estimated_R(x);//这个是qci
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    //如果没有足够的旋转，最后一个奇异值将较大，又因为奇异值从大到小排列，所以取出了倒数第二个奇异值，大于0.25才认为有效
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}
/**
 * @brief 利用对极约束求解两帧图像的旋转矩阵
 * 
 * @param corres 
 * @return Matrix3d 求得的旋转矩阵，是R12
 */
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        //SVD分解本质矩阵E
        decomposeE(E, R1, R2, t1, t2);
        //以下是选取得到的4组R和t
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;
        //解出来R21

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);//转置为R12
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}
/**
 * @brief 通过三角化检查R t是否合理
 * 
 * @param l l相机的观测
 * @param r r相机的观测
 * @param R 旋转矩阵
 * @param t 平移向量
 * @return double 
 */
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;//保存三角化之后的地图点，每一列是一个点的坐标
    //第一帧设置为单位阵
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    //第二帧设置为R t对应的位姿
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    //三角化特征点为地图点，参考坐标系为l图片（世界坐标系）
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        //由于齐次坐标，所以要取出第四维并归一化
        double normal_factor = pointcloud.col(i).at<float>(3);
        //得到各点在各自相机坐标系下的坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        //通过深度是否大于0判断是否合理
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;//根据比例作为评判标准
}
/**
 * @brief 分解本质矩阵E，参考多视图几何//?
 * 
 * @param E 
 * @param R1 
 * @param R2 
 * @param t1 
 * @param t2 
 */
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
