#ifndef RANSACLIB_EXAMPLE_CALIBRATED_ABSOLUTE_POSE_ESTIMATOR_QPEP_H_
#define RANSACLIB_EXAMPLE_CALIBRATED_ABSOLUTE_POSE_ESTIMATOR_QPEP_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace ransac_lib {

namespace calibrated_absolute_pose_QPEP {
using Eigen::Vector3d;
typedef Eigen::Matrix<double, 3, 4> CameraPose;
typedef std::vector<CameraPose, Eigen::aligned_allocator<CameraPose>>
    CameraPoses;

typedef std::vector<Eigen::Vector2d> Points2D;
typedef std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>> Points3D;

class CalibratedAbsolutePoseEstimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CalibratedAbsolutePoseEstimator(const double f_x, const double f_y,
                                  const double squared_inlier_threshold,
                                  const Points2D& points2D,
                                  const Points3D& points3D);

  inline int min_sample_size() const { return 4; }

  inline int non_minimal_sample_size() const { return 6; }

  inline int num_data() const { return num_data_; }

  int MinimalSolver(const std::vector<int>& sample, CameraPoses* poses) const;


  int NonMinimalSolver(const std::vector<int>& sample, CameraPose* pose) const;

  double EvaluateModelOnPoint(const CameraPose& pose, int i) const;

  void LeastSquares(const std::vector<int>& sample, CameraPose* pose) const;

 protected:
  // Focal lengths in x- and y-directions.
  double focal_x_;
  double focal_y_;
  double squared_inlier_threshold_;
  // Matrix holding the 2D point positions.
  Points2D points2D_;
  // Matrix holding the corresponding 3D point positions.
  Points3D points3D_;
  int num_data_;

};

}  // namespace calibrated_absolute_pose

}  // namespace ransac_lib

#endif  // RANSACLIB_EXAMPLE_CALIBRATED_ABSOLUTE_POSE_ESTIMATOR_QPEP_H_
