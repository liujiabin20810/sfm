// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 SOLVEPNPLIB_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// SOLVEPNPLIB_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。

#include <vector>
#include <opencv2/core/core.hpp>

#ifdef SOLVEPNPLIB_EXPORTS
#define SOLVEPNPLIB_API __declspec(dllexport)
#else
#define SOLVEPNPLIB_API __declspec(dllimport)
#endif
#pragma once
#include "Common.h"
using namespace bundler;

#define MIN_INLIERS_EST_PROJECTION 6

SOLVEPNPLIB_API bool FindAndVerifyCamera(int num_points, 
	std::vector<cv::Point3f> points3d,
	std::vector<cv::Point2f> points2d,
	std::vector<int> idxs_solve,
	cv::Mat& K, cv::Mat& R, cv::Mat& t, 
	double proj_estimation_threshold,
	double proj_estimation_threshold_weak,
	std::vector<int> &inliers,
	std::vector<int> &inliers_weak,
	std::vector<int> &outliers);

SOLVEPNPLIB_API bool RefineCameraAndPoints(int num_points, 
	std::vector<cv::Point3f> points3d,	std::vector<cv::Point2f> points2d,
	std::vector<int> idxs_solve,camera_params_t& camera,
	std::vector<ImageKeyVector> &pt_views,std::vector<int> &inliers);

SOLVEPNPLIB_API int Triangulate(std::vector<cv::Point2f> l_pt,std::vector<cv::Point2f> r_pt,int num_points,
	int matches[][2],camera_params_t l_camera, camera_params_t r_camera,std::vector<cv::Point3f>& points ,std::vector<int>& indexs);

SOLVEPNPLIB_API bool Triangulate(cv::Point2f l_pt, cv::Point2f r_pt,
	camera_params_t l_camera, camera_params_t r_camera,cv::Point3f & point);

SOLVEPNPLIB_API void InitializeCameraParams(camera_params_t &camera);

SOLVEPNPLIB_API void GetIntrinsics(const camera_params_t &camera, double *K);

SOLVEPNPLIB_API double GetCameraDistance(camera_params_t *c1, camera_params_t *c2);

SOLVEPNPLIB_API cv::Point2f UndistortNormPoint(cv::Point2f p, camera_params_t c);

SOLVEPNPLIB_API bool EstimatePose(std::vector<cv::Point2f> keys1, std::vector<cv::Point2f> keys2,
	int match_num,int matches[][2],camera_params_t &c1, camera_params_t &c2);