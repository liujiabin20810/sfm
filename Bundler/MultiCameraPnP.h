#ifndef  BUNDLER_MULTICAMERAPNP_HPP__
#define BUNDLER_MULTICAMERAPNP_HPP__
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <map>
#include <set>
#include <fstream>

#include "Common.h"
#include "FeatureMatcher.h"
#include "MatchTracks.h"
#include "BundleAdjuster.h"

namespace bundler
{
	
	class CMultiCameraPnP : public CPruneFeature
	{
		
	public:

		CMultiCameraPnP(char outpath[], std::vector<std::string> images_list,std::vector<cv::Mat> images) : CPruneFeature(images_list,images)
		{
			strcpy(m_path,outpath);
		};

		// set calibration matrix
		void initCalibMatrix(cv::Mat _K, cv::Mat _distortion_coeff);

		// Calculate all images match
		bool match();
		/**
		* Get an initial 3D point cloud from 2 views only
		*/
		bool GetBaseLineTriangulation(std::vector<ImageKeyVector>& point_views);

		void RecoverDepthFromImages();


		cv::Mat drawImageMatches(int _index_i, int _index_j);

		~CMultiCameraPnP();

	private:
		/* Setup the initial camera pair for bundle adjustment */
		int SetupInitialCameraPair(int i_best, int j_best, std::vector<ImageKeyVector>& pt_view);

		/* Pick a good initial pair of cameras to bootstrap the bundle
	     * adjustment */
		void PickInitialPair(int &i_best, int &j_best);

		int GetNumTrackMatches(int img1, int img2);

		Keypoint& GetImageKey(int img, int key);

		std::vector<cv::Point2f> GetImageKeypoints(int image);

		bool FindCameraMatrices(cv::Mat F,cv::Matx34d& P,cv::Matx34d& P1, cv::Point2f p1,cv::Point2f p2);
		
		bool FindCameraMatrices(cv::Mat F,camera_params_t& camera1, camera_params_t& camera2,
				cv::Point2f kp1,cv::Point2f kp2);

		std::vector<int> ProjectMeanError(camera_params_t  camera,std::vector<cv::Point3f> points3d,std::vector<cv::Point2f> points2d);

		double TriangulatePoints(int ith_camera,int jth_camera,camera_params_t& camera1, camera_params_t& camera2,std::vector<PointData>& pointcloud);

		bool TriangulatePointsBetweenViews(int older_view,int working_view,camera_params_t& camera1, camera_params_t& camera2,std::vector<ImageKeyVector>& pt_view );

		void SetBundleAdjustData(std::vector<CameraT> &camera_data,std::vector<Point3D>& point_data,
			std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx,
			std::vector<ImageKeyVector> pt_views);

		void GetBundleAdjustData(std::vector<CameraT> &camera_data,std::vector<Point3D>& point_data,
			std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx);

		void AdjustCurrentBundle(std::vector<ImageKeyVector> pt_views, bool _debug);

		// check the point_view is correct or not.
		int CheckPointKeyConsistency(const std::vector<ImageKeyVector> pt_views,
			std::vector<int> added_order);

		void DumpOutputFile(const char *output_dir, const char *filename, 
			int num_points,	std::vector<int> order, 
			camera_params_t *cameras, 
			std::vector<PointData>& points,
			std::vector<ImageKeyVector> &pt_views);

		void Bundler2PMVS(std::vector<ImageKeyVector> pt_view);

		//////////////////////////////////////////////////////////////////////////
		bool FindCameraMatrices(int ith_camera,int jth_camera,
			cv::Matx34d& P,
			cv::Matx34d& P1);

		int SelectPMatrix(std::vector<cv::Matx34d> _4Pmatrixs,cv::Point3d u,cv::Point3d u1);
		int SelectPMatrix(std::vector<cv::Matx34d> _4Pmatrixs,cv::Point2f u,cv::Point2f u1);

	
		double TriangulatePoints(
			int ith_camera,
			int jth_camera,
			std::vector<PointData>& pointcloud);


		cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
			cv::Matx34d P,		//camera 1 matrix
			cv::Point3d u1,		//homogenous image point in 2nd camera
			cv::Matx34d P1		//camera 2 matrix
			);

		cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
			cv::Matx34d P,			//camera 1 matrix
			cv::Point3d u1,			//homogenous image point in 2nd camera
			cv::Matx34d P1			//camera 2 matrix
			);

		bool FindPoseEstimation(int working_view,
				cv::Mat_<double>& t,cv::Mat_<double>& R ,std::vector<ImageKeyVector>& pt_view ); 

		void Find2D3DCorrespondences(int working_view, 
			std::vector<cv::Point3f>& ppcloud, 
			std::vector<cv::Point2f>& imgPoints,
			std::vector<ImageKeyVector> pt_view,
			std::vector<int> & keys_solve,
			std::vector<int> & idxs_solve);


		void WriteCloudPoint(char filename[],int start_idx);

		void SaveModelFile(const char* outpath,std::vector<ImageKeyVector> pt_views);

	public:

		std::map<int, cv::Matx34d> Pmats;
		std::vector<CloudPoint> PointCloud;

		std::vector<PointData> m_point_data; /* Information about 3D
					    * points in the scene */
		int m_point_data_index;
	private:

		char m_path[256];

		//BundleAdjuster mba;

		camera_params_t *m_cameras;
		int curr_num_cameras;
		float m_f,m_cx,m_cy,m_r;

		//////////////////////////////////////////////////////////////////////////
		std::map<int,cv::Mat> camera_matrixs_inv;
		std::map<int,cv::Mat> camera_matrixs;
		//////////////////////////////////////////////////////////////////////////

		int m_initial_pair[2];       /* Images to use as the initial pair
				  * during bundle adjustment */

		int m_first_view;
		int m_second_view; //baseline's second view other to 0
		std::vector<int> done_views;
		std::vector<int> good_views;

		MatchTracks m_matchTracks;
	};

}


#endif