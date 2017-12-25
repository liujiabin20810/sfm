// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� SOLVEPNPLIB_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// SOLVEPNPLIB_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�

#include <vector>
#include <opencv2/core/core.hpp>

#ifdef SOLVEPNPLIB_EXPORTS
#define SOLVEPNPLIB_API __declspec(dllexport)
#else
#define SOLVEPNPLIB_API __declspec(dllimport)
#endif

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