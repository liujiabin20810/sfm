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

namespace bundler
{
	
	class CMultiCameraPnP : public CFeatureMatcher
	{
		
	public:

		CMultiCameraPnP(std::vector<std::string> images_list,std::vector<cv::Mat> images) : CFeatureMatcher(images_list,images){};

		// set calibration matrix
		void initCalibMatrix(cv::Mat _K, cv::Mat _distortion_coeff);

		// Calculate all images match
		bool match();

		/**
		* Get an initial 3D point cloud from 2 views only
		*/
		bool GetBaseLineTriangulation();

		void RecoverDepthFromImages();


		~CMultiCameraPnP();

	private:
		
		bool FindCameraMatrices(const cv::Mat& K,
			const cv::Mat& Kinv,
			const cv::Mat& distcoeff,
			const cv::vector<cv::KeyPoint>& imgpts1,
			const cv::vector<cv::KeyPoint>& imgpts2,
			int ith_camera,
			int jth_camera,
			cv::Mat F,
			cv::Matx34d& P,
			cv::Matx34d& P1);

		bool TriangulatePointsBetweenViews(int working_view,int older_view );

		double TriangulatePoints(
			int ith_camera,
			int jth_camera,
			std::vector<PointData>& pointcloud);

		void AdjustCurrentBundle();

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

		bool FindPoseEstimation(int working_view,cv::Mat_<double>& rvec,
				cv::Mat_<double>& t,cv::Mat_<double>& R); 

		void Find2D3DCorrespondences(int working_view, 
			std::vector<cv::Point3f>& ppcloud, 
			std::vector<cv::Point2f>& imgPoints);

		void WriteCloudPoint();

	public:

		std::map<int, cv::Matx34d> Pmats;
		std::vector<CloudPoint> PointCloud;

		std::vector<PointData> m_point_data; /* Information about 3D
					    * points in the scene */
		int m_point_data_index;
	private:

		cv::Mat K, Kinv, distortion_coeff;
		cv::Mat distcoeff_32f;
		cv::Mat K_32f;

		int m_first_view;
		int m_second_view; //baseline's second view other to 0
		std::set<int> done_views;
		std::set<int> good_views;

		MatchTracks m_matchTracks;
	};

}


#endif