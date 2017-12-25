/*
 *  BundleAdjuster.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/18/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */
#ifndef  BUNDLER_ADJUSTER_HPP__
#define BUNDLER_ADJUSTER_HPP__
#pragma once

#include <set>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.h"
#include "pba.h"

namespace bundler
{
	class BundleAdjuster {
	public:

		BundleAdjuster();

		~BundleAdjuster();

		void SetCameraData(std::vector<CameraT>& camera_data);

		void SetPointData(std::vector<Point3D>& point_data);

		void SetProjection(std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx);
		
		void RunBundleAdjustment(std::vector<CameraT>& camera_data,std::vector<Point3D>& point_data,
			std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx);

// 		void adjustBundle(std::vector<CloudPoint>& pointcloud, 
// 						  cv::Mat& cam_matrix,
// 						  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
// 						  std::map<int ,cv::Matx34d>& Pmats);
// 
// 		void adjustBundle(std::vector<PointData>& pointcloud, 
// 						  cv::Mat& cam_matrix,
// 						  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
// 						  std::map<int ,cv::Matx34d>& Pmats, std::set<int> good_views);

	private:
		 
	
		 //CameraT, Point3D, Point2D are defined in src/pba/DataInterface.h
// 		 std::vector<CameraT>        _camera_data;    //camera (input/ouput)
// 		 std::vector<Point3D>        _point_data;     //3D point(iput/output)
// 		 std::vector<Point2D>        _measurements;   //measurment/projection vector
// 		 std::vector<int>            _camidx, ptidx;  //index of camera/point for each projection

	};
}

#endif