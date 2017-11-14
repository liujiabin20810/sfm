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

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.h"

namespace bundler
{
	class BundleAdjuster {
	public:
		void adjustBundle(std::vector<CloudPoint>& pointcloud, 
						  cv::Mat& cam_matrix,
						  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
						  std::map<int ,cv::Matx34d>& Pmats);

		void adjustBundle(std::vector<PointData>& pointcloud, 
						  cv::Mat& cam_matrix,
						  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
						  std::map<int ,cv::Matx34d>& Pmats);
	private:
		int Count2DMeasurements(const std::vector<CloudPoint>& pointcloud);
		int Count2DMeasurements(const std::vector<PointData>& pointcloud);
	};
}

#endif