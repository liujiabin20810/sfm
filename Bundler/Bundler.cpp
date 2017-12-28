// Bundler.cpp : 定义控制台应用程序的入口点。
//
#pragma once
#include "stdafx.h"
#include <io.h>
#include <direct.h>
#include <windows.h>
#include "SIFTGPU.hpp"

#include "Common.h"
#include "MultiCameraPnP.h"
#include "MatchTracks.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vector>
#include <iostream>
#include <fstream>

#pragma comment(lib, "pba.lib")
#pragma comment(lib, "siftgpu.lib")
#pragma comment(lib,"opencv_calib3d249.lib")
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")
#pragma comment(lib,"opencv_features2d249.lib")
#pragma comment(lib,"solvePnPlib.lib")

using namespace cv;
using namespace std;
using namespace bundler;


int _tmain(int argc, _TCHAR* argv[])
{
	char sourcePath[256] = "../../data/photo";
	char maskPath[256] = "../../data/photo";
	char cameraPath[256] = "";
	char outPath[256] = "../../basecloud";
	if(argc > 4)
	{
		strcpy(sourcePath,argv[1]);
		strcpy(maskPath,argv[2]);
		strcpy(cameraPath,argv[3]);
		strcpy(outPath,argv[4]);
	}

	Utils m_utils;
	// load source images
	int n_images = m_utils.open_imgs_dir(string(sourcePath));
	if (n_images < 0)
	{
		std::cerr<<"load source image error.\n";
		return -1;
	}

	CFeatureMatcher feature(m_utils.m_imageNameList,m_utils.m_images);

	feature.calc_sift_feature();

	//load mask images
	n_images = m_utils.open_imgs_dir(string(maskPath));
	if(n_images < 0 )
	{
		std::cerr<<"load mask image error.\n";
		return -1;
	}
	m_utils.loadCalibMatrix(cameraPath);
	
	CMultiCameraPnP fmatcher(outPath,feature.m_images_list,m_utils.m_images);
	fmatcher.initCalibMatrix(m_utils.m_K,m_utils.m_distortion_coeff);

	if(!fmatcher.match())
		return -1;

	double t1 = getTickCount();

	fmatcher.RecoverDepthFromImages();

	double t2 = getTickCount();

	double t = (t2 -t1)/getTickFrequency();

	std::cout<<"process time :"<<t<<" s"<<std::endl;

	return 0;


}

