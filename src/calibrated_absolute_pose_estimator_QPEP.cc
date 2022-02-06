#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/StdVector>

#include "calibrated_absolute_pose_estimator_QPEP.h"
#include <LibQPEP/utils.h>
#include <LibQPEP/pnp_WQD.h>
#include <LibQPEP/solver_WQ_1_2_3_4_5_9_13_17_33_49_approx.h>
#include <LibQPEP/QPEP_grobner.h>
#include <LibQPEP/QPEP_lm_single.h>
#include <LibQPEP/misc_pnp_funcs.h>

namespace ransac_lib {

    namespace calibrated_absolute_pose_QPEP {

        struct NormalizedReprojectionError {
            NormalizedReprojectionError(double x, double y, double X, double Y, double Z,
                                        double fx, double fy)
                    : point2D_x(x),
                      point2D_y(y),
                      point3D_X(X),
                      point3D_Y(Y),
                      point3D_Z(Z),
                      f_x(fx),
                      f_y(fy) {}

            template<typename T>
            bool operator()(const T *const camera, T *residuals) const {
                T pp[3];
                pp[0] = (T) point3D_X;
                pp[1] = (T) point3D_Y;
                pp[2] = (T) point3D_Z;

                T pp_rot[3];
                ceres::AngleAxisRotatePoint(camera, pp, pp_rot);
                pp_rot[0] += camera[3];
                pp_rot[1] += camera[4];
                pp_rot[2] += camera[5];

                double c_x = 320.0;
                double c_y = 160.0;
                T ppp_rot[3];
                ppp_rot[0] = ((T)f_x) * pp_rot[0] + ((T)c_x) * pp_rot[2];
                ppp_rot[1] = ((T)f_y) * pp_rot[1] + ((T)c_y) * pp_rot[2];
                ppp_rot[2] = pp_rot[2];

                T x_proj = ppp_rot[0] / ppp_rot[2];
                T y_proj = ppp_rot[1] / ppp_rot[2];

                residuals[0] = static_cast<T>(point2D_x) - x_proj;
                residuals[1] = static_cast<T>(point2D_y) - y_proj;

                return true;
            }

            // Factory function
            static ceres::CostFunction *CreateCost(const double x, const double y,
                                                   const double X, const double Y,
                                                   const double Z, const double fx,
                                                   const double fy) {
                return (new ceres::AutoDiffCostFunction<NormalizedReprojectionError, 2, 6>(
                        new NormalizedReprojectionError(x, y, X, Y, Z, fx, fy)));
            }

            double point2D_x;
            double point2D_y;
            double point3D_X;
            double point3D_Y;
            double point3D_Z;
            double f_x;
            double f_y;
        };

        CalibratedAbsolutePoseEstimator::CalibratedAbsolutePoseEstimator(
                const double f_x, const double f_y, const double squared_inlier_threshold,
                const Points2D &points2D, const Points3D &points3D)
                : focal_x_(f_x),
                  focal_y_(f_y),
                  squared_inlier_threshold_(squared_inlier_threshold),
                  points2D_(points2D),
                  points3D_(points3D) {
            num_data_ = static_cast<int>(points2D_.size());
        }

        const double problem_scale = 1e-8;

        Eigen::Matrix<double, 4, 24> coef_f_q_sym_;
        Eigen::Matrix<double, 3, 10> coefs_tq_;
        Eigen::Matrix<double, 3, 3> pinvG_;

        int CalibratedAbsolutePoseEstimator::MinimalSolver(const std::vector<int> &sample, CameraPoses *poses) const {

            Eigen::Matrix<double, 4, 64> W;
            Eigen::Matrix<double, 4, 4> Q;
            Eigen::Matrix<double, 3, 37> D;
            Eigen::Matrix<double, 1, 70> coef_J_pure;

            Eigen::Matrix3d K;
            K.setZero();
            K(0, 0) = focal_x_;
            K(0, 2) = 320;
            K(1, 1) = focal_y_;
            K(1, 2) = 160;
            K(2, 2) = 1.0;
            std::vector<Eigen::Vector2d> image_pt;
            std::vector<Eigen::Vector3d> world_pt;
            for (int i = 0; i < sample.size(); ++i) {
                image_pt.push_back(points2D_[sample[i]]);
                world_pt.push_back(points3D_[sample[i]]);
            }
            pnp_WQD(W, Q, D, coef_f_q_sym_, coef_J_pure, coefs_tq_, pinvG_, image_pt, world_pt, K, problem_scale);

            Eigen::Matrix<double, 3, 64> W_ = W.topRows(3);
            Eigen::Matrix<double, 3, 4> Q_ = Q.topRows(3);
            W_.row(0) = W.row(0) + W.row(1) + W.row(2);
            W_.row(1) = W.row(1) + W.row(2) + W.row(3);
            W_.row(2) = W.row(2) + W.row(3) + W.row(0);

            Q_.row(0) = Q.row(0) + Q.row(1) + Q.row(2);
            Q_.row(1) = Q.row(1) + Q.row(2) + Q.row(3);
            Q_.row(2) = Q.row(2) + Q.row(3) + Q.row(0);

            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            Eigen::Matrix4d X;
            X.setZero();
            double min[27];
            struct QPEP_options opt;
            opt.ModuleName = "solver_WQ_1_2_3_4_5_9_13_17_33_49_approx";
            opt.DecompositionMethod = "PartialPivLU";

            struct QPEP_runtime stat = QPEP_WQ_grobner(R, t, X, min, W_, Q_,
                                                       reinterpret_cast<solver_func_handle>(solver_WQ_1_2_3_4_5_9_13_17_33_49_approx),
                                                       reinterpret_cast<mon_J_pure_func_handle>(mon_J_pure_pnp_func),
                                                       reinterpret_cast<t_func_handle>(t_pnp_func),
                                                       coef_J_pure, coefs_tq_, pinvG_, nullptr, opt);
            bool flag = true;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (std::isnan(X(i, j))) {
                        flag = false;
                        return flag;
                    }

            CameraPose pose_;
            pose_.topLeftCorner<3, 3>() = X.topLeftCorner<3, 3>();
            pose_(0, 3) = X(0, 3);
            pose_(1, 3) = X(1, 3);
            pose_(2, 3) = X(2, 3);
            poses->push_back(pose_);

            return flag;
        }


        int CalibratedAbsolutePoseEstimator::NonMinimalSolver(const std::vector<int> &sample, CameraPose *pose) const {
            CameraPoses poses;
            MinimalSolver(sample, &poses);
            *pose = poses[0];
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            Eigen::Matrix4d X;
            R = (*pose).topLeftCorner<3, 3>();

            Eigen::Vector4d q0 = R2q(R);
            struct QPEP_runtime stat = QPEP_lm_single(R, t, X, q0, 1000, 5e-2,
                                                      reinterpret_cast<eq_Jacob_func_handle>(eq_Jacob_pnp_func),
                                                      reinterpret_cast<t_func_handle>(t_pnp_func),
                                                      coef_f_q_sym_, coefs_tq_, pinvG_, stat);
            bool flag = true;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (std::isnan(X(i, j))) {
                        flag = false;
                        return flag;
                    }

            CameraPose pose_;
            pose_.topLeftCorner<3, 3>() = R;
            pose_(0, 3) = t.x();
            pose_(1, 3) = t.y();
            pose_(2, 3) = t.z();
            std::memcpy(pose, &pose_, sizeof(CameraPose));

//            Eigen::Matrix<double, 3, 4> pose_inv = poses[0];
//            LeastSquares(sample, &pose_inv);
//            std::cout << "LS Pose: " << std::endl;
//            std::cout << pose_inv << std::endl;
            return flag;
        }


// Evaluates the pose on the i-th data point.
        double CalibratedAbsolutePoseEstimator::EvaluateModelOnPoint(
                const CameraPose &pose, int i) const {
            Eigen::Vector3d p_c =
                    pose.topLeftCorner<3, 3>() * points3D_[i] + pose.col(3);
            Eigen::Matrix3d K;
            K << focal_x_, 0, 320.0,
                 0, focal_y_, 160.0,
                 0, 0, 1.0;
            p_c = K * p_c;

            // Check whether point projects behind the camera.
            if (p_c[2] < 0.0) return std::numeric_limits<double>::max();

            Eigen::Vector2d p_2d = p_c.head<2>() / p_c[2];
            return (p_2d - points2D_[i]).squaredNorm() * 1e-2;
        }

// Reference implementation using Ceres for refinement.
        void CalibratedAbsolutePoseEstimator::LeastSquares(
                const std::vector<int> &sample, CameraPose *pose) const {
            Eigen::AngleAxisd aax(pose->topLeftCorner<3, 3>());
            Eigen::Vector3d aax_vec = aax.axis() * aax.angle();
            double camera[6];
            camera[0] = aax_vec[0];
            camera[1] = aax_vec[1];
            camera[2] = aax_vec[2];
            camera[3] = pose->col(3)[0];
            camera[4] = pose->col(3)[1];
            camera[5] = pose->col(3)[2];

            ceres::Problem refinement_problem;
//  ceres::LossFunction* cauchy_loss = new ceres::CauchyLoss(1.0);
            const int kSampleSize = static_cast<int>(sample.size());
            for (int i = 0; i < kSampleSize; ++i) {
                const int kIdx = sample[i];
                const Eigen::Vector2d &p_img = points2D_[kIdx];
                const Eigen::Vector3d &p_3D = points3D_[kIdx];
                ceres::CostFunction *cost_function =
                        NormalizedReprojectionError::CreateCost(
                                p_img[0], p_img[1], p_3D[0], p_3D[1], p_3D[2], focal_x_, focal_y_);
//     double error_i = std::sqrt(EvaluateModelOnPoint(*pose, sample[i]));
//     error_i = std::max(0.00001, error_i);
                refinement_problem.AddResidualBlock(cost_function, nullptr, camera);

            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            options.function_tolerance = 1e-8;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &refinement_problem, &summary);

//            std::cout << summary.BriefReport() << std::endl;
//  delete cauchy_loss;
//  cauchy_loss = nullptr;

            if (summary.IsSolutionUsable()) {
                Eigen::Vector3d axis(camera[0], camera[1], camera[2]);
                double angle = axis.norm();
                axis.normalize();
                aax.axis() = axis;
                aax.angle() = angle;

                pose->topLeftCorner<3, 3>() = aax.toRotationMatrix();
                pose->col(3) = Eigen::Vector3d(camera[3], camera[4], camera[5]);
            }
        }

    }  // namespace calibrated_absolute_pose

}  // namespace ransac_lib
