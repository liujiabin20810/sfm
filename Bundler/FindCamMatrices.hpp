#ifndef BUNDLER_FINDCAMMATRICES_HPP__
#define BUNDLER_FINDCAMMATRICES_HPP__

#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>
#include <list>
#include <set>

#include "SiftGPU.h"
namespace bundler
{
	class FindCamMatrices
	{
	public:
		//det(R) = 1 Õý½»¾ØÕó
		bool CheckCoherentRotation(cv::Mat_<double>& R);

		//Try to eliminate keypoints based on the fundamental matrix
		//(although this is not the proper way to do this)
		void GetCameraMat(cv::Mat K,cv::Mat F, cv::Matx34d &P);

		cv::Mat_<double> LinearLSTriangulation(
			cv::Point3d u,//homogenous image point (u,v,1)
			cv::Matx34d P,//camera 1 matrix
			cv::Point3d u1,//homogenous image point in 2nd camera
			cv::Matx34d P1//camera 2 matrix
			);

		double TriangulatePoints(
			std::vector<cv::Point2f> pt_set1,
			std::vector<cv::Point2f> pt_set2,
			const cv::Mat K,
			const cv::Mat Kinv,
			const cv::Matx34d& P,
			const cv::Matx34d& P1,
			std::vector<cv::Point3d>& pointcloud);

		FindCamMatrices(void);

		~FindCamMatrices(void);

	private:

		struct CloudPoint {
			cv::Point3d pt;
			std::vector<int>index_of_2d_origin;
		};

	};
}


#endif


