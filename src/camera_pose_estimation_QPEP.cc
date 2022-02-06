#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <RansacLib/ransac.h>
#include "calibrated_absolute_pose_estimator_QPEP.h"

namespace ransac_lib {
    namespace calibrated_absolute_pose_QPEP {

        int generateProjectedPoints(std::vector<Eigen::Vector2d> &image_pt,
                                     std::vector<double> &s,
                                     std::vector<Eigen::Vector3d> &world_pt,
                                     const Eigen::Matrix3d &K,
                                     const Eigen::Matrix3d &R,
                                     const Eigen::Vector3d &t,
                                     const double width, const double height,
                                     const int num_inliers,
                                     const int num_outliers, double inlier_threshold,
                                     const double min_depth, const double max_depth) {
            int numPoints = num_inliers + num_outliers;;
            s.resize(numPoints);
            image_pt.resize(numPoints);
            world_pt.resize(numPoints);

            std::vector<int> indices(numPoints);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rand_dev;
            std::mt19937 rng(rand_dev());
            std::shuffle(indices.begin(), indices.end(), rng);
            std::uniform_real_distribution<double> distr(-1.0, 1.0);
            const double kWidthHalf = width * 0.5;
            const double kHeightHalf = height * 0.5;
            std::uniform_real_distribution<double> distr_x(-kWidthHalf, kWidthHalf);
            std::uniform_real_distribution<double> distr_y(-kHeightHalf, kHeightHalf);

            int iter = 0;
            for (int i = 0; i < numPoints; ++i) {
                while(true) {
                    ++iter;
                    if(iter > numPoints * 50) {
                        return -1;
                    }
                    Eigen::Vector3d world_pt_ = Eigen::Vector3d(distr(rng), distr(rng), distr(rng));
                    Eigen::Vector4d world_point(world_pt_.x(), world_pt_.y(), world_pt_.z(), 1);
                    Eigen::Matrix<double, 3, 4> cameraMatrix;
                    cameraMatrix << R.transpose(), t;
                    cameraMatrix = K * cameraMatrix;
                    Eigen::Vector3d projectedPoint = cameraMatrix * world_point;
                    s[i] = projectedPoint(2);
                    if(min_depth < s[i] && s[i] < max_depth)
                    {
                        Eigen::Vector2d projectedPoint_(projectedPoint(0), projectedPoint(1));
                        Eigen::Vector2d true_pt = projectedPoint_ / s[i];
                        image_pt[i] = true_pt + 1e-5 * Eigen::Vector2d(distr_x(rng), distr_y(rng)) * inlier_threshold;
                        if(i >= num_inliers)
                        {
                            Eigen::Vector2d tmp = Eigen::Vector2d(distr_x(rng), distr_y(rng));
                            image_pt[i] += 1e0 * tmp;
//                            std::cout << "Generating outliers" << std::endl;
//                            std::cout << tmp << std::endl;
                        }
                        world_pt[i] = world_pt_;
                        break;
                    }
                }
            }

            return 0;
        }

        void GenerateRandomInstance(const double width, const double height,
                                    const double focal_length, const int num_inliers,
                                    const int num_outliers, double inlier_threshold,
                                    const double min_depth, const double max_depth,
                                    Points2D *points2D,
                                    Points3D *points3D) {
            const int kNumPoints = num_inliers + num_outliers;
            points2D->resize(kNumPoints);
            points3D->resize(kNumPoints);

            Eigen::Quaterniond q(Eigen::Vector4d::Random());
            q.normalize();

            std::vector<Eigen::Vector2d> image_pt;
            std::vector<Eigen::Vector3d> world_pt;
            std::vector<double> s;

            int res = -1;
            while(res == -1)
            {
                Eigen::Matrix3d R = q.toRotationMatrix();
                Eigen::Vector3d t = Eigen::Vector3d::Random();
                Eigen::Matrix3d K;
                K << focal_length, 0, width / 2,
                        0.0, focal_length, height / 2,
                        0.0, 0.0, 1.0;

                res = generateProjectedPoints(image_pt, s, world_pt, K, R, t,
                                        width, height, num_inliers, num_outliers,
                                        inlier_threshold, min_depth, max_depth);
                if(res != -1)
                {
                    Eigen::Matrix<double, 3, 4> tmp;
                    tmp << R, t;
                    std::cout << "True X: " << std::endl << tmp << std::endl;
                }
            }

            for(int i = 0; i < kNumPoints; ++i)
            {
                (*points2D)[i] = image_pt[i];
                (*points3D)[i] = world_pt[i];
            }
        }

    }  // namespace calibrated_absolute_pose
}  // namespace ransac_lib


int main(int argc, char **argv) {
    ransac_lib::LORansacOptions options;
    options.min_num_iterations_ = 100u;
    options.max_num_iterations_ = 100000u;

    std::random_device rand_dev;
    options.random_seed_ = rand_dev();
    srand((unsigned int) time(0));

    const int kNumDataPoints = 2000;

//    std::vector<double> outlier_ratios = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
//                                          0.6, 0.7, 0.8, 0.9, 0.95};

    std::vector<double> outlier_ratios = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

    const double kWidth = 640.0;
    const double kHeight = 320.0;
    const double kFocalLength = (kWidth * 0.5) / std::tan(60.0 * M_PI / 180.0);
    const double kInThreshPX = 12.0;

    options.squared_inlier_threshold_ = kInThreshPX * kInThreshPX;
    std::cout.precision(25);

    for (const double outlier_ratio: outlier_ratios) {
        std::cout << " Inlier ratio: " << 1.0 - outlier_ratio << std::endl;
        int num_outliers =
                static_cast<int>(static_cast<double>(kNumDataPoints) * outlier_ratio);
        int num_inliers = kNumDataPoints - num_outliers;

        ransac_lib::calibrated_absolute_pose_QPEP::Points2D points2D;
        ransac_lib::calibrated_absolute_pose_QPEP::Points3D points3D;
        ransac_lib::calibrated_absolute_pose_QPEP::GenerateRandomInstance(
                kWidth, kHeight, kFocalLength, num_inliers, num_outliers, 2.0, 2.0,
                10.0, &points2D, &points3D);

//        std::cout << "   ... instance generated" << std::endl;

        ransac_lib::calibrated_absolute_pose_QPEP::CalibratedAbsolutePoseEstimator
                solver(kFocalLength, kFocalLength, kInThreshPX * kInThreshPX, points2D,
                       points3D);

//        std::vector<int> samp_;
//        for (int i = 0; i < kNumDataPoints; ++i)
//            samp_.push_back(i);
//        ransac_lib::calibrated_absolute_pose_QPEP::CameraPoses pose;
//        ransac_lib::calibrated_absolute_pose_QPEP::CameraPoses *pose_ptr = &pose;
//        solver.MinimalSolver(samp_, pose_ptr);
//        std::cout << std::endl << "QPEP Initial Pose: " << std::endl;
//        std::cout << pose_ptr[0][0] << std::endl;
//
//        ransac_lib::calibrated_absolute_pose_QPEP::CameraPose pose_;
//        ransac_lib::calibrated_absolute_pose_QPEP::CameraPose *pose_ptr_ = &pose_;
//        solver.NonMinimalSolver(samp_, pose_ptr_);
//        std::cout << std::endl << "QPEP Refine Pose: " << std::endl;
//        std::cout << pose_ptr_[0] << std::endl;

        // Runs LO-MSAC, as described in Lebeda et al., BMVC 2012.
//        std::cout << "   ... running LO-MSAC" << std::endl;
        {
            options.threshold_multiplier_ = 10.0;
            options.min_sample_multiplicator_ = 7;
            options.num_lsq_iterations_ = 4;
            options.num_lo_steps_ = 10;
            options.final_least_squares_ = true;
            options.random_seed_ = (unsigned int) time(0);
            options.max_num_iterations_ = 1000;

            ransac_lib::LocallyOptimizedMSAC<
                    ransac_lib::calibrated_absolute_pose_QPEP::CameraPose,
                    ransac_lib::calibrated_absolute_pose_QPEP::CameraPoses,
                    ransac_lib::calibrated_absolute_pose_QPEP::CalibratedAbsolutePoseEstimator>
                    lomsac;
            ransac_lib::RansacStatistics ransac_stats;
            auto ransac_start = std::chrono::system_clock::now();
            ransac_lib::calibrated_absolute_pose_QPEP::CameraPose best_model;
            int num_ransac_inliers =
                    lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);
            auto ransac_end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = ransac_end - ransac_start;

            Eigen::Matrix<double, 3, 4> tmp;
            std::cout << "Best Fit:" << std::endl;
            tmp << best_model.topLeftCorner<3, 3>().transpose(),
                    best_model.col(3);
            std::cout << tmp << std::endl << std::endl << std::endl;

            std::cout << "   ... LOMSAC found " << num_ransac_inliers
                      << " inliers in " << ransac_stats.num_iterations
                      << " iterations with an inlier ratio of "
                      << ransac_stats.inlier_ratio << std::endl;
            std::cout << "   ... LOMSAC (inliers ratio:" << 1.0 - outlier_ratio << ") took " << elapsed_seconds.count() << " s"
                      << std::endl;

        }
    }
    return 0;
}
