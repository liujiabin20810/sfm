/*
 *  BundleAdjuster.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/18/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "BundleAdjuster.h"


#define V3DLIB_ENABLE_SUITESPARSE

using namespace std;
using namespace cv; 

//count number of 2D measurements

namespace bundler
{


	BundleAdjuster::BundleAdjuster()
	{
	
	}
	
	BundleAdjuster::~BundleAdjuster()
	{

	}

	void  BundleAdjuster::SetCameraData(std::vector<CameraT>& camera_data)
	{
		 //m_pba.SetCameraData(camera_data.size(),  &camera_data[0]); //set camera parameters
	}

	void BundleAdjuster::SetPointData(std::vector<Point3D>& point_data)
	{

		//m_pba.SetPointData(point_data.size(),&point_data[0]); //set 3D point data

	}

	void BundleAdjuster::SetProjection(std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx)
	{
		//m_pba.SetProjection(measurements.size(), &measurements[0], &ptidx[0], &camidx[0]);//set the projections
	}

	void BundleAdjuster::RunBundleAdjustment(std::vector<CameraT>& camera_data,std::vector<Point3D>& point_data,
		std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx)
	{
		static ParallelBA m_pba;
		m_pba.GetInternalConfig()-> __use_radial_distortion = -1;
		m_pba.GetInternalConfig()->__verbose_level = 1;
		m_pba.SetCameraData(camera_data.size(),  &camera_data[0]);
		m_pba.SetPointData(point_data.size(),&point_data[0]); //set 3D point data
		m_pba.SetProjection(measurements.size(), &measurements[0], &ptidx[0], &camidx[0]);//set the projections
		m_pba.RunBundleAdjustment();
	}

}