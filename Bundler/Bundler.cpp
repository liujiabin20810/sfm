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

bool hasEnding (std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

bool hasEndingLower (string const &fullString_, string const &_ending)
{
	string fullstring = fullString_, ending = _ending;
	transform(fullString_.begin(),fullString_.end(),fullstring.begin(),::tolower); // to lower
	return hasEnding(fullstring,ending);
}

void open_imgs_dir(char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names, double downscale_factor) {
	if (dir_name == NULL) {
		return;
	}

	string dir_name_ = string(dir_name);
	vector<string> files_;

#ifndef WIN32
	//open a directory the POSIX way

	DIR *dp;
	struct dirent *ep;     
	dp = opendir (dir_name);

	if (dp != NULL)
	{
		while (ep = readdir (dp)) {
			if (ep->d_name[0] != '.')
				files_.push_back(ep->d_name);
		}

		(void) closedir (dp);
	}
	else {
		cerr << ("Couldn't open the directory");
		return;
	}

#else
	//open a directory the WIN32 way
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA fdata;

	if(dir_name_[dir_name_.size()-1] == '\\' || dir_name_[dir_name_.size()-1] == '/') {
		dir_name_ = dir_name_.substr(0,dir_name_.size()-1);
	}

	hFind = FindFirstFile(string(dir_name_).append("\\*").c_str(), &fdata);	
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strcmp(fdata.cFileName, ".") != 0 &&
				strcmp(fdata.cFileName, "..") != 0)
			{
				if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					continue; // a diretory
				}
				else
				{
					files_.push_back(fdata.cFileName);
				}
			}
		}
		while (FindNextFile(hFind, &fdata) != 0);
	} else {
		cerr << "can't open directory\n";
		return;
	}

	if (GetLastError() != ERROR_NO_MORE_FILES)
	{
		FindClose(hFind);
		cerr << "some other error with opening directory: " << GetLastError() << endl;
		return;
	}

	FindClose(hFind);
	hFind = INVALID_HANDLE_VALUE;
#endif

	for (unsigned int i=0; i<files_.size(); i++) {
		if (files_[i][0] == '.' || !(hasEndingLower(files_[i],"jpg")||hasEndingLower(files_[i],"png"))) {
			continue;
		}
		cv::Mat m_ = cv::imread(string(dir_name_).append("/").append(files_[i]));
		if(downscale_factor != 1.0)
			cv::resize(m_,m_,Size(),downscale_factor,downscale_factor);
		images_names.push_back(files_[i]);
		images.push_back(m_);
	}


}

int findImage(char strPath[],vector<string> & imgList)
{
	if (_chdir(strPath) != 0)
	{
		std::cout<<"the image directory: "<<string(strPath)<<" not exist..."<<std::endl;
		return 0;
	}

	//如果目录的最后一个字母不是'\',则在最后加上一个'\'
	int len=strlen(strPath);
	if (strPath[len-1] != '\\')
		strcat(strPath,"\\");

	long hFile;
	_finddata_t fileinfo;
	if ((hFile=_findfirst("*.jpg",&fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果不是,则进行处理
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				char filename[_MAX_PATH];
				//				strcpy(filename,strPath);
				strcpy(filename,fileinfo.name);
				//				printf("%s\n",filename);
				imgList.push_back(string(filename));
			}
		} while (_findnext(hFile,&fileinfo) == 0);
		_findclose(hFile);
	}

	return imgList.size();
}

int _tmain(int argc, _TCHAR* argv[])
{
	char strPath[1024] = "../feet/source";

	if(argc > 1)
		strcpy(strPath,argv[1]);

// 	vector<string> list;
// 	int n = findImage(strPath,list);
// 
// 	cout << "Read Images Num: "<< n << endl;
// 	for(int i=0; i< n; i++)
// 		cout<<list[i]<<endl;

	Utils m_utils;
	int n_images = m_utils.open_imgs_dir(string(strPath));
	m_utils.loadCalibMatrix(strPath);

	if (n_images < 0)
		return -1;

	CMultiCameraPnP fmatcher(m_utils.m_imageNameList,m_utils.m_images);

	fmatcher.initCalibMatrix(m_utils.m_K,m_utils.m_distortion_coeff);

	//fmatcher.read_sift_feature();

	if(!fmatcher.match())
		return -1;

	double t1 = getTickCount();

	fmatcher.RecoverDepthFromImages();

	double t2 = getTickCount();

	double t = (t2 -t1)/getTickFrequency();

	
	std::cout<<"process time :"<<t<<" s"<<std::endl;
// 	bundler::SIFT sift(4096);
// 
// 	sift.setImageList(list);
// 
// 	vector<vector<SiftKeypoint>> vkeypoints;
// 	vector<vector<float>> vdescriptors;
// 
// 	sift(vkeypoints,vdescriptors);
// 
// 	vector<vector<pair<int,int>>> match_buffer;
// 	sift.match(vkeypoints,vdescriptors,match_buffer);
// 
// 	cout<<"image num: "<<sift.m_nimage<<" match num: "<<match_buffer.size()<<endl;
// 
// 	ofstream fp_match("match.txt");
// 
// 	sift.writeSiftMatch(fp_match,match_buffer);
// 
// 	fp_match.close();
// 	ifstream fin_match("match.txt");
// 	sift.readSiftMatch(fin_match,match_buffer);
// 	fin_match.close();
// 
// 	for(int i=0; i < match_buffer.size(); i++)
// 	{
// 		vector<pair<int,int>> buff = match_buffer[i];
// 		cout<<i<<": "<<buff.size()<<endl;
// 
// 		if( i == 54)
// 		{
// 			for (int k=0; k < buff.size(); k++)
// 			{
// 				cout<< buff[k].first<<" "<< buff[k].second<<endl;
// 			}
// 		}
// 	}

	return 0;


}

