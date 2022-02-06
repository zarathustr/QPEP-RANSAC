cmake_minimum_required (VERSION 3.0)

if (EXISTS "${CMAKE_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
endif ()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -std=c++17")

find_package (Eigen3 REQUIRED)
find_package (Ceres REQUIRED)
find_package (LibQPEP REQUIRED)

include_directories (
  ${CMAKE_SOURCE_DIR}
  ${EIGEN_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${LibQPEP_INCLUDE_DIRS}
  ../../RansacLib
  ../RansacLib
  ../
)


add_executable (camera_pose_estimation_QPEP
        camera_pose_estimation_QPEP.cc
        calibrated_absolute_pose_estimator_QPEP.cc
        calibrated_absolute_pose_estimator_QPEP.h)
target_link_libraries (camera_pose_estimation_QPEP ${CERES_LIBRARIES} ${LibQPEP_LIBRARIES})