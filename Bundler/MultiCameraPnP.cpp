#pragma once
#include "MultiCameraPnP.h"
#include "solvePnPlib.h"
#include "util.h"

#include <windows.h>
#include <direct.h>
#include <io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <list>
#include <fstream>


#define USE_BUNDLER_EPOSE
#define EPSILON 0.0001

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

namespace bundler
{

	Utils::Utils()
	{
		m_images.clear();
		m_imageNameList.clear();
	}

	Utils::~Utils()
	{

	}

	bool Utils::hasEnding(std::string const &fullString, std::string const &ending)
	{
		if (fullString.length() >= ending.length()) {
			return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
		}
		else {
			return false;
		}
	}

	bool Utils::hasEndingLower(std::string const &fullString_, std::string const &_ending)
	{
		std::string fullstring = fullString_, ending = _ending;
		transform(fullString_.begin(), fullString_.end(), fullstring.begin(), ::tolower); // to lower
		return hasEnding(fullstring, ending);
	}

	int Utils::open_imgs_dir(std::string dir_name_)
	{
		if (dir_name_.empty()) {
			return -1;
		}

		m_imageNameList.clear();
		m_images.clear();

		std::vector<std::string> files_;

#ifndef WIN32
		//open a directory the POSIX way

		DIR *dp;
		struct dirent *ep;
		dp = opendir(dir_name);

		if (dp != NULL)
		{
			while (ep = readdir(dp)) {
				if (ep->d_name[0] != '.')
					files_.push_back(ep->d_name);
			}

			(void)closedir(dp);
		}
		else {
			cerr << ("Couldn't open the directory");
			return;
		}

#else
		//open a directory the WIN32 way
		HANDLE hFind = INVALID_HANDLE_VALUE;
		WIN32_FIND_DATA fdata;

		if (dir_name_[dir_name_.size() - 1] == '\\' || dir_name_[dir_name_.size() - 1] == '/') {
			dir_name_ = dir_name_.substr(0, dir_name_.size() - 1);
		}

		//std::cout << "Path : " << dir_name_ << std::endl;
		hFind = FindFirstFile(std::string(dir_name_).append("\\*").c_str(), &fdata);
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
			} while (FindNextFile(hFind, &fdata) != 0);
		}
		else {
			std::cerr << "can't open directory\n";
			return -1;
		}

		if (GetLastError() != ERROR_NO_MORE_FILES)
		{
			FindClose(hFind);
			std::cerr << "some other error with opening directory: " << GetLastError() << std::endl;
			return -1;
		}

		FindClose(hFind);
		hFind = INVALID_HANDLE_VALUE;
#endif
		for (unsigned int i = 0; i < files_.size(); i++)
		{

			if (files_[i][0] == '.' || !(hasEndingLower(files_[i], "jpg") || hasEndingLower(files_[i], "bmp"))) {
				continue;
			}
			std::string filename = std::string(dir_name_).append("/").append(files_[i]);

			cv::Mat m_ = cv::imread(filename);
			if (m_.empty())
			{
				std::cerr << "Open image " << filename  << " err." << std::endl;
				return -1;
			}
			m_imageNameList.push_back(filename);
			m_images.push_back(m_);
		}

		m_image_width = m_images[0].cols;
		m_image_height = m_images[0].rows;


		return m_imageNameList.size();
	}

	int Utils::findImage(char strPath[])
	{
		if (_chdir(strPath) != 0)
		{
			std::cout << "the image directory: " << std::string(strPath) << " not exist..." << std::endl;
			return -1;
		}

		//如果目录的最后一个字母不是'\',则在最后加上一个'\'
		int len = strlen(strPath);
		if (strPath[len - 1] != '\\')
			strcat(strPath, "\\");

		long hFile;
		_finddata_t fileinfo;
		if ((hFile = _findfirst("*.jpg", &fileinfo)) != -1)
		{
			do
			{
				//检查是不是目录
				//如果不是,则进行处理
				if (!(fileinfo.attrib & _A_SUBDIR))
				{
					char filename[_MAX_PATH];
					//strcpy(filename,strPath);
					strcpy(filename, fileinfo.name);

					m_imageNameList.push_back(std::string(filename));
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}

		int n = m_imageNameList.size();

		for (int i = 0; i < n; i++)
		{
			cv::Mat m_ = cv::imread(m_imageNameList[i], 1);
			if (m_.empty())
			{
				std::cerr << "Open image " << m_imageNameList[i] << " err." << std::endl;
				return -1;
			}

			m_images.push_back(m_);
		}

		return n;
	}

	int Utils::loadCalibMatrix(char strPath[])
	{
		//load calibration matrix
		cv::Mat cam_matrix;

		cv::FileStorage fs;
		if (fs.open(std::string(strPath) + "\\camera_data.yml", cv::FileStorage::READ)) {
			fs["camera_matrix"] >> cam_matrix;
			fs["distortion_coefficients"] >> m_distortion_coeff;
		}
		else {
			//no calibration matrix file - mockup calibration

			double max_w_h = MAX(m_image_height, m_image_width);
			//iphone 6 1/3"  4.80mm*3.60mm , 4.2mm
			//crazyhorse (4.9/6.1)
			double focal_length_in_pixel = max_w_h * (4.2/4.8);
			cam_matrix = (cv::Mat_<double>(3, 3) << focal_length_in_pixel, 0, m_image_width / 2.0,
				0, focal_length_in_pixel, m_image_height / 2.0,
				0, 0, 1);
			m_distortion_coeff = cv::Mat_<double>::zeros(1, 4);
		}

		m_K = cam_matrix;
		cv::invert(m_K, m_Kinv); //get inverse of camera matrix

		m_distortion_coeff.convertTo(m_distcoeff_32f, CV_32FC1);
		m_K.convertTo(m_K_32f, CV_32FC1);

		return 0;
	}

	void Utils::WriteBundleFile(const char *bundle_file, 
		const std::vector<camera_params_t> &cameras,
		std::vector<int> added_order,
		const std::vector<PointData> &points)
	{
		FILE *f = fopen(bundle_file, "w");
		if (f == NULL) {
			printf("Error opening file %s for reading\n", bundle_file);
			return;
		}

		int num_images = cameras.size();
		int num_points = points.size();

		/* Count the number of good images */
		int num_good_images = 0;
		int *map = new int[num_images];
		for (int i = 0; i < num_images; i++) {
			if (cameras[i].f == 0) {
				map[i] = -1;                
				continue;
			}

			map[i] = num_good_images;
			num_good_images++;
		}

		printf("[WriteBundleFile] Writing %d images and %d points...\n",
			num_good_images, num_points);

		fprintf(f, "# Bundle file v0.3\n"); // Bundle Version
		fprintf(f, "%d %d\n", num_good_images, num_points);    

		/* Write cameras */
		for (int i = 0; i < num_images; i++) {
			if (cameras[i].f == 0)
				continue;

			/* Focal length */
			fprintf(f, "%lf 0.0 0.0\n", cameras[i].f);

			/* Rotation */
			const double *R = cameras[i].R;
			/* Translation */
			const double *c = cameras[i].t;
			double t[3]; 
			t[0] = -(R[0]*c[0] + R[1]*c[1] + R[2]*c[2]);
			t[1] = -(R[3]*c[0] + R[4]*c[1] + R[5]*c[2]);
			t[2] = -(R[6]*c[0] + R[7]*c[1] + R[8]*c[2]);
			
			fprintf(f, "%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n", 
				R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
			fprintf(f, "%lf %lf %lf\n", t[0], t[1], t[2]);
		}

		std::vector<int> pt_view_nums(num_good_images,0);
		/* Write points */
		for (int i = 0; i < num_points; i++) {
			/* Position */
			const double *pos = points[i].m_pos;
			fprintf(f, "%lf %lf %lf\n", pos[0], pos[1], pos[2]);

			/* Color */
			const int *color = points[i].m_color;
			fprintf(f, "%d %d %d\n", 
				(color[0]), (color[1]), (color[2]));

			int num_visible;
			num_visible = points[i].m_views.size();
			fprintf(f, "%d", num_visible);

			for (int j = 0; j < num_visible; j++) {
				int view = map[points[i].m_views[j].image];
				assert(view >= 0 && view < num_good_images);
				int key =  points[i].m_views[j].key;
				double x = - (points[i].m_views[j].x);
				double y = - (points[i].m_views[j].y);

				pt_view_nums[view] += 1;
				fprintf(f, " %d %d %0.2f %0.2f", view, key, x, y);
			}

			fprintf(f, "\n");
		}
//		for(int i = 0 ; i < num_good_images; i++)
//			fprintf(f,"%d %d,",i,pt_view_nums[i]);

		fprintf(f, "\n");
		fclose(f);

		delete [] map;
	}

	void Utils::WritePMVS(const char *output_path, 
		std::vector<std::string> images, 
		std::vector<camera_params_t> &cameras,
		std::vector<int> added_order,
		const std::vector<PointData> &points)
	{
		char buf[2048];
		sprintf(buf, "%s/bundle.rd.out", output_path);    
		WriteBundleFile(buf,cameras,added_order,points);

		//////////////////////////////////////////////////////////////////////////
		
		//visualize
		//txt
		int num_files = (int) images.size();

		sprintf(buf, "%s/list.txt", output_path);
		FILE *f = fopen(buf, "w");

		sprintf(buf,"%s/visualize/",output_path);
		char buf2[256];
		sprintf(buf2,"%s/txt/",output_path);
		_mkdir(buf);
		_mkdir(buf2);

		int count = 0;
		for (int i =0 ; i < num_files; i++)
		{
 			if (cameras[i].f == 0.0)
				continue;
			cv::Mat im = cv::imread(images[i],1);
			if(im.empty())
				continue;
			// list.txt
			int pos = images[i].rfind(".");
			char outbuf[128];
			sprintf(outbuf,"visualize/%08d",count);
			std::string  filename = std::string(outbuf) + images[i].substr(pos);
			fprintf(f, "%s\n", filename.c_str());
			//////////////////////////////////////////////////////////////////////////
			// images
			sprintf(outbuf,"%08d",count);
			std::string out = std::string(buf) + std::string(outbuf) + images[i].substr(pos);
			std::cout<< "save image :"<<out<<std::endl;
			cv::imwrite(out,im);
			//////////////////////////////////////////////////////////////////////////
			// txt
			char camerafile[256];
			sprintf(camerafile, "%s%08d.txt",buf2, count);
			FILE *camf = fopen(camerafile, "w");
			assert(camf);

			/* Compute the projection matrix */
			double focal = cameras[i].f;
			double *R = cameras[i].R;
//		    double *t = cameras[i].t;
			double t[3]; 

			t[0] = -(R[0]*cameras[i].t[0] + R[1]*cameras[i].t[1] + R[2]*cameras[i].t[2]);
			t[1] = -(R[3]*cameras[i].t[0] + R[4]*cameras[i].t[1] + R[5]*cameras[i].t[2]);
			t[2] = -(R[6]*cameras[i].t[0] + R[7]*cameras[i].t[1] + R[8]*cameras[i].t[2]);

			int w = im.cols;
			int h = im.rows;
			
			double K[9] = 
			{-focal, 0.0, 0.0,
			0.0, focal, 0.0,
			0.0, 0.0, 1.0 };

			double Ptmp[12] = 
			{ R[0], R[1], R[2], t[0],
			R[3], R[4], R[5], t[1],
			R[6], R[7], R[8], t[2] };

			cv::Mat Kmat = cv::Mat(3,3,CV_64F,K);
			cv::Mat Ptmat = cv::Mat(3,4,CV_64F,Ptmp);

			cv::Mat_<double> Pmat = -Kmat*Ptmat;

// 			matrix_product(3, 3, 3, 4, K, Ptmp, P);
// 			matrix_scale(3, 4, P, -1.0, P);

			fprintf(camf, "CONTOUR\n");
			fprintf(camf, "%0.6f %0.6f %0.6f %0.6f\n", Pmat(0,0), Pmat(0,1), Pmat(0,2),  Pmat(0,3));
			fprintf(camf, "%0.6f %0.6f %0.6f %0.6f\n", Pmat(1,0), Pmat(1,1), Pmat(1,2),  Pmat(1,3));
			fprintf(camf, "%0.6f %0.6f %0.6f %0.6f\n", Pmat(2,0), Pmat(2,2), Pmat(2,3),  Pmat(2,3));

			fclose(camf);
			++count;
		}

		fclose(f);
	}

	void CMultiCameraPnP::initCalibMatrix(cv::Mat _K, cv::Mat _distortion_coeff)
	{
		int num = n_images;

		m_cameras = new camera_params_t[num];

		m_f = 0.5* (_K.at<double>(0,0) + _K.at<double>(1,1));

		m_cx = _K.at<double>(0,2);
		m_cy = _K.at<double>(1,2);
	
		m_r = _distortion_coeff.at<double>(0,0);

		curr_num_cameras = 0;
		m_point_data_index = 0;

		m_initial_pair[0] = m_initial_pair[1] = -1;
	}

	cv::Mat_<double> CMultiCameraPnP::LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
		cv::Matx34d P,		//camera 1 matrix
		cv::Point3d u1,		//homogenous image point in 2nd camera
		cv::Matx34d P1		//camera 2 matrix
		)
	{
		//build matrix A for homogenous equation system Ax = 0
		//assume X = (x,y,z,1), for Linear-LS method
		//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
		//	cout << "u " << u <<", u1 " << u1 << endl;
		//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
		//	A(0) = u.x*P(2)-P(0);
		//	A(1) = u.y*P(2)-P(1);
		//	A(2) = u.x*P(1)-u.y*P(0);
		//	A(3) = u1.x*P1(2)-P1(0);
		//	A(4) = u1.y*P1(2)-P1(1);
		//	A(5) = u1.x*P(1)-u1.y*P1(0);
		//	Matx43d A; //not working for some reason...
		//	A(0) = u.x*P(2)-P(0);
		//	A(1) = u.y*P(2)-P(1);
		//	A(2) = u1.x*P1(2)-P1(0);
		//	A(3) = u1.y*P1(2)-P1(1);

		cv::Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
			u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
			u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
			u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
			);

		cv::Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
			-(u.y*P(2, 3) - P(1, 3)),
			-(u1.x*P1(2, 3) - P1(0, 3)),
			-(u1.y*P1(2, 3) - P1(1, 3)));

		cv::Mat_<double> X;
		cv::solve(A, B, X, cv::DECOMP_SVD);

		return X;
	}


	/**
	From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
	*/
	cv::Mat_<double> CMultiCameraPnP::IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
		cv::Matx34d P,			//camera 1 matrix
		cv::Point3d u1,			//homogenous image point in 2nd camera
		cv::Matx34d P1			//camera 2 matrix
		) {
		double wi = 1, wi1 = 1;
		cv::Mat_<double> X(4, 1);
		for (int i = 0; i<10; i++)
		{ //Hartley suggests 10 iterations at most
			
			cv::Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
			
			X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

			
			//recalculate weights
			double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
			double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

			//breaking point
			if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON)
				break;

			wi = p2x;
			wi1 = p2x1;

			//reweight equations and solve
			cv::Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
				(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
				(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
				(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
				);
			
			cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
				-(u.y*P(2, 3) - P(1, 3)) / wi,
				-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
				-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
				);

			cv::solve(A, B, X_, cv::DECOMP_SVD);

			X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
			
		}

		return X;
	}

	double CMultiCameraPnP::TriangulatePoints(
			int ith_camera,
			int jth_camera,
			std::vector<PointData>& pointcloud)
	{
		pointcloud.clear();

		cv::Matx34d P = Pmats[ith_camera];
		cv::Matx34d P1 = Pmats[jth_camera];

		std::vector<cv::Point2f> pt_set1 = GetImageKeypoints(ith_camera);
		std::vector<cv::Point2f> pt_set2 = GetImageKeypoints(jth_camera);

		MatchIndex idx = GetMatchIndex(ith_camera,jth_camera);
		std::vector<KeypointMatch>& matchList = m_matchTracks.m_matches_table.GetMatchList(idx);
		
		ImageData& img_data = m_matchTracks.m_image_data[ith_camera];

		unsigned int match_size = matchList.size();
		

		cv::vector<double> reproj_error;
#ifdef USE_OPENCV_TRIGUL

		std::vector<int> queryIdxs,trainIdxs;
		//std::vector<cv::Point2f> set_pt1,set_pt2;
		cv::Mat set_pt1 = cv::Mat(match_size,1,CV_32FC2);
		cv::Mat set_pt2 = cv::Mat(match_size,1,CV_32FC2);

		std::vector<cv::Point2f> vec_pt1,vec_pt2;
		for(int i = 0; i < match_size; i++)
		{
			int queryIdx = matchList[i].m_idx1;

			int track_idx = GetImageKey(ith_camera,queryIdx).m_track;
			int pt3d_idx = m_matchTracks.m_track_data[track_idx].m_extra;

			// this track_data used , skip it!
			if(pt3d_idx >= 0)
				continue;

			cv::Point2f kp = pt_set1[queryIdx];
			queryIdxs.push_back(queryIdx);
			vec_pt1.push_back(kp);

			set_pt1.at<cv::Vec2f>(i)[0] = kp.x;
			set_pt1.at<cv::Vec2f>(i)[1] = kp.y;

			int trainIdx = matchList[i].m_idx2;
			cv::Point2f kp1 = pt_set2[trainIdx];			
			trainIdxs.push_back(trainIdx);
			vec_pt2.push_back(kp1);

			set_pt2.at<cv::Vec2f>(i)[0] = kp1.x;
			set_pt2.at<cv::Vec2f>(i)[1] = kp1.y;
		}

		cv::Mat undis_pt1 = cv::Mat(2,match_size,CV_32FC1);
		cv::Mat undis_pt2 = cv::Mat(2,match_size,CV_32FC1);
		float focal_length = K_32f.at<float>(0,0) ;
		float cx = K_32f.at<float>(0,2);
		float cy = K_32f.at<float>(1,2);
	
		std::cout<<focal_length<<" "<<cx<<" "<<cy<<std::endl;

		for (int i = 0 ; i < match_size; i++)
		{
			
			double ux1,ux2;
			double uy1,uy2;
			double uz1,uz2;

			ux1 = 1.0/focal_length*(vec_pt1[i].x - cx);
			uy1 = 1.0/focal_length*(vec_pt1[i].y - cy);
			uz1 = 1.0;

			ux2 = 1.0/focal_length*(vec_pt2[i].x - cx);
			uy2 = 1.0/focal_length*(vec_pt2[i].y - cy);
			uz2 = 1.0;

			undis_pt1.ptr<float>(0)[i] = ux1/uz1;
			undis_pt1.ptr<float>(1)[i] = uy1/uz1;

			undis_pt2.ptr<float>(0)[i] = ux2/uz2;
			undis_pt2.ptr<float>(1)[i] = uy2/uz2;

		}
		
		std::cout<<"[undistortPoints] "<<vec_pt1[0]<<" "<<undis_pt1.col(0)<<std::endl;

		cv::Mat pt_4d_h;
		//  pt_4d_h out , 4 row , N col
		cv::triangulatePoints(P,P1,undis_pt1,undis_pt2,pt_4d_h);

		std::cout<<pt_4d_h.rows<<" "<<pt_4d_h.cols<<std::endl;
		std::cout<<"Point 4d:"<<pt_4d_h.colRange(0,2)<<std::endl;
		std::vector<cv::Point3f> pt_3d;
		cv::Mat pt_4d_t = pt_4d_h.t(); // reshape 
		cv::Mat pt_4d_h_shape = (pt_4d_t).reshape(4,1);

		std::cout<<"Point 4d:"<<pt_4d_h_shape.colRange(0,2)<<std::endl;

		convertPointsFromHomogeneous(pt_4d_h_shape,pt_3d);
// 		for (int i =0; i < match_size; i++)
// 		{
// 			cv::Mat coord4f = pt_4d_h.col(i);
// 			if(i == 0 )
// 				std::cout<<"Point 4d:"<<coord4f<<std::endl;
// 			float x , y ,z;
// 			 x = y = z = 0.0;
// 			
// 			 float w = coord4f.at<float>(3);
// 			
// 			if(w != 0.0)
// 			{
// 				x = coord4f.at<float>(0)/w;
// 				y = coord4f.at<float>(1)/w;
// 				z = coord4f.at<float>(2)/w;
// 			}
// 
// 			pt_3d.push_back(cv::Point3f(x,y,z));
// 		}

		std::cout<<"Point :"<<pt_3d[0]<<std::endl;
		cv::Mat_<double> R1 = (cv::Mat_<double>(3,3) << P1(0,0),P1(0,1),P1(0,2), 
														P1(1,0),P1(1,1),P1(1,2), 
														P1(2,0),P1(2,1),P1(2,2));
		cv::Vec3d rvec; 
		Rodrigues(R1 ,rvec);
		cv::Vec3d tvec(P1(0,3),P1(1,3),P1(2,3));
		std::vector<cv::Point2f> reprojected_pt_set2;

		projectPoints(pt_3d,rvec,tvec,K,distortion_coeff,reprojected_pt_set2);

		std::cout<<"src : "<<vec_pt2[0]<<std::endl;
		std::cout<<"re-project: "<<reprojected_pt_set2[0]<<std::endl;
		for(int i = 0; i < match_size; i++ )
		{
			double reprj_err = cv::norm(reprojected_pt_set2[i] - vec_pt2[i]);
			{
				reproj_error.push_back(reprj_err);
				
				PointData cp;
				cp.m_pos[0] = pt_3d[i].x;
				cp.m_pos[1] = pt_3d[i].y;
				cp.m_pos[2] = pt_3d[i].z;
				
				//cp.m_views.push_back(ImageKey(ith_camera,queryIdxs[i]));
				//cp.m_views.push_back(MatchIndex(jth_camera,trainIdxs[i]));
				cp.m_ref_image = ith_camera;
				int x = int(set_pt1.at<cv::Vec2f>(i)[0] + 0.5);
				int y = int(set_pt1.at<cv::Vec2f>(i)[1] + 0.5);

				cp.m_color[0] = (int)GetImageKey(ith_camera,queryIdxs[i]).m_b;
				cp.m_color[1] = (int)GetImageKey(ith_camera,queryIdxs[i]).m_g;
				cp.m_color[2] = (int)GetImageKey(ith_camera,queryIdxs[i]).m_r;

				pointcloud.push_back(cp);
			}
		}

#else

		std::ofstream fp3d("pt3d.txt");
		if(!fp3d.is_open())
			return -1;

		fp3d << ith_camera <<" "<< jth_camera << std::endl;

		fp3d << P  << std::endl;
		fp3d << P1 << std::endl;
		cv::Mat_<double> KP1 = camera_matrixs[jth_camera] * cv::Mat(P1);

		//#pragma omp parallel for num_threads(4)		
		for (int i = 0; i < match_size; i++) {
			
			int trainIdx = matchList[i].m_idx2;
			int queryIdx = matchList[i].m_idx1;
			cv::Point2f kp1 = pt_set2[trainIdx];
			cv::Point2f kp = pt_set1[queryIdx];
			int track_idx = GetImageKey(jth_camera,trainIdx).m_track;
			int pt3d_idx = m_matchTracks.m_track_data[track_idx].m_extra;
			// this track_data used , skip it!
			if(pt3d_idx >= 0)
			{
				//continue;
				fp3d<<jth_camera<<" "<<trainIdx<<"; "<<ith_camera<<" "<<queryIdx<<" "<<std::endl;
				fp3d<<kp1<<" "<<kp<<std::endl;
				fp3d<<i<<" ["<<m_point_data[pt3d_idx].m_pos[0]<<" "<<m_point_data[pt3d_idx].m_pos[1]<<" "
					<<m_point_data[pt3d_idx].m_pos[2]<<"] [";
			}
			//int trainIdx = dmatch[i].trainIdx;
			
			cv::Point3d u1(kp1.x, kp1.y, 1.0);
			cv::Mat_<double> um1 = camera_matrixs_inv[jth_camera] * cv::Mat_<double>(u1);
			u1 = um1.at<cv::Point3d>(0);
			/////////////////////////////////////////////

			
			//int queryIdx = dmatch[i].queryIdx;
			
			cv::Point3d u(kp.x, kp.y, 1.0);
			cv::Mat_<double> um = camera_matrixs_inv[ith_camera] * cv::Mat_<double>(u);
			u = um.at<cv::Point3d>(0);
			
			cv::Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);
			cv::Mat_<double> xPt_img = KP1 * X;
			if(pt3d_idx >= 0)
			{
				fp3d<<X(0)<<" "<<X(1)<<" "<<X(2)<<" "<<X(3)<<"] "<<X(0)-m_point_data[pt3d_idx].m_pos[0]<<std::endl;
				continue;
			}
			cv::Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

			//#pragma omp critical
			{
				double reprj_err = cv::norm(xPt_img_ - kp1);
				
				if(i == 0)
				{
					std::cout<<"2d point: "<<kp<<" "<<kp1<<std::endl;
					std::cout<<"3d point: "<<X(0)<<" "<<X(1)<<" "<<X(2)<<std::endl;
					std::cout<<"re-projection err: "<<reprj_err<< std::endl;
				}

				reproj_error.push_back(reprj_err);

				PointData cp;
				cp.m_pos[0] = X(0);
				cp.m_pos[1] = X(1);
				cp.m_pos[2] = X(2);

				view_t v1,v2;
				v1.image = ith_camera;
				v1.key = queryIdx;
				v1.x = GetImageKey(ith_camera,queryIdx).m_x;
				v1.y = GetImageKey(ith_camera,queryIdx).m_y;

				v2.image = jth_camera;
				v2.key = trainIdx;
				v2.x = GetImageKey(jth_camera,trainIdx).m_x;
				v2.y = GetImageKey(jth_camera,trainIdx).m_y;
				cp.m_views.push_back(v1);
				cp.m_views.push_back(v2);

				cp.m_ref_image = ith_camera;

				cp.m_color[0] = (int)GetImageKey(ith_camera,queryIdx).m_b;
				cp.m_color[1] = (int)GetImageKey(ith_camera,queryIdx).m_g;
				cp.m_color[2] = (int)GetImageKey(ith_camera,queryIdx).m_r;
				
				pointcloud.push_back(cp);

			}
		}

#endif
		fp3d.close();

		if(reproj_error.empty())
			return -1.0;

		double minErr = 100, maxErr = 0;
		int minIdx,maxIdx;
		for(int i=0; i < reproj_error.size(); i++)
		{
			//std::cout<<reproj_error[i]<<" ";
			//if(i > 0 && i % 100 == 0 )
			//	std::cout<<std::endl;
			if(minErr > reproj_error[i]){ minErr = reproj_error[i]; minIdx = i;}
			if(maxErr < reproj_error[i]){ maxErr = reproj_error[i]; maxIdx = i;}
			
		}

		std::cout<<"[TriangulatePoints]  minErr = "<<minErr<<" maxErr = "<<maxErr<<std::endl;
		cv::Scalar me = cv::mean(reproj_error);

		std::cout<<"[TriangulatePoints] meanErr =  "<<me[0]<<std::endl;
		//std::cout<<"[TriangulatePoints] minIdx =  "<<minIdx<<" maxIdx = "<<maxIdx<<std::endl;

		return me[0];
	}

	double CMultiCameraPnP::TriangulatePoints(int ith_camera,int jth_camera,camera_params_t& camera1, camera_params_t& camera2,std::vector<PointData>& pointcloud)
	{
		pointcloud.clear();

		cv::Matx34d P(camera1.R[0],camera1.R[1],camera1.R[2],camera1.t[0],
					  camera1.R[3],camera1.R[4],camera1.R[5],camera1.t[1],
					  camera1.R[6],camera1.R[7],camera1.R[8],camera1.t[2]);

		cv::Matx34d P1(camera2.R[0],camera2.R[1],camera2.R[2],camera2.t[0],
					   camera2.R[3],camera2.R[4],camera2.R[5],camera2.t[1],
					   camera2.R[6],camera2.R[7],camera2.R[8],camera2.t[2]);


		double K[9],K1[9];
		GetIntrinsics(camera1,K);
		GetIntrinsics(camera2,K1);

		cv::Mat mK = cv::Mat(3,3,CV_64F,K);
		cv::Mat mK1 = cv::Mat(3,3,CV_64F,K1);

		std::vector<cv::Point2f> pt_set1 = GetImageKeypoints(ith_camera);
		std::vector<cv::Point2f> pt_set2 = GetImageKeypoints(jth_camera);

		MatchIndex idx = GetMatchIndex(ith_camera,jth_camera);
		std::vector<KeypointMatch>& matchList = m_matchTracks.m_matches_table.GetMatchList(idx);
		ImageData& img_data = m_matchTracks.m_image_data[ith_camera];
		unsigned int match_size = matchList.size();

		cv::Mat_<double> KP1 = mK1 * cv::Mat(P1);
		cv::vector<double> reproj_error;

		for (int i = 0; i < match_size; i++) {

			int trainIdx = matchList[i].m_idx2;
			int queryIdx = matchList[i].m_idx1;
			
			cv::Point2f kp1 = pt_set2[trainIdx];
			cv::Point2f kp = pt_set1[queryIdx];

			cv::Point3d u1(kp1.x-m_cx, kp1.y-m_cy, -1.0);
			cv::Mat_<double> um1 = mK1.inv() * cv::Mat_<double>(u1);
			u1 = -um1.at<cv::Point3d>(0);
			/////////////////////////////////////////////

			cv::Point3d u(kp.x-m_cx, kp.y - m_cy, -1.0);
			cv::Mat_<double> um = mK.inv() * cv::Mat_<double>(u);
			u = -um.at<cv::Point3d>(0);

			cv::Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);
			cv::Mat_<double> xPt_img = KP1 * X;

			cv::Point2f xPt_img_(xPt_img(0) / xPt_img(2) - m_cx, xPt_img(1) / xPt_img(2) - m_cy);

			//#pragma omp critical
			{
				double reprj_err = cv::norm(xPt_img_ + kp1);

				if(reprj_err > 100)
					continue;

				if(i == 0)
				{
					std::cout<<"2d point: "<<kp<<" "<<kp1<<std::endl;
					std::cout<<"3d point: "<<X(0)<<" "<<X(1)<<" "<<X(2)<<std::endl;
					std::cout<<"projection point: "<<xPt_img(0)/xPt_img(2)<<" "<<xPt_img(1)/xPt_img(2)<<std::endl;
					std::cout<<"re-projection err: "<<reprj_err<< std::endl;
				}

				reproj_error.push_back(reprj_err);

				PointData cp;
				cp.m_pos[0] = X(0);
				cp.m_pos[1] = X(1);
				cp.m_pos[2] = X(2);

				view_t v1,v2;
				v1.image = ith_camera; v1.key = queryIdx;
				v1.x = kp.x; v1.y = kp.y;

				v2.image = jth_camera;  v2.key = trainIdx;
				v2.x = kp1.x; 	v2.y = kp.y;

				cp.m_views.push_back(v1);
				cp.m_views.push_back(v2);
				cp.m_ref_image = ith_camera;

				cp.m_color[0] = (int)GetImageKey(ith_camera,queryIdx).m_b;
				cp.m_color[1] = (int)GetImageKey(ith_camera,queryIdx).m_g;
				cp.m_color[2] = (int)GetImageKey(ith_camera,queryIdx).m_r;

				pointcloud.push_back(cp);

			}
		}


		if(reproj_error.empty())
			return -1.0;

		double minErr = 100, maxErr = 0;
		int minIdx,maxIdx;
		for(int i=0; i < reproj_error.size(); i++)
		{
			//std::cout<<reproj_error[i]<<" ";
			//if(i > 0 && i % 100 == 0 )
			//	std::cout<<std::endl;
			if(minErr > reproj_error[i]){ minErr = reproj_error[i]; minIdx = i;}
			if(maxErr < reproj_error[i]){ maxErr = reproj_error[i]; maxIdx = i;}

		}

		std::cout<<"[TriangulatePoints]  minErr = "<<minErr<<" maxErr = "<<maxErr<<std::endl;
		cv::Scalar me = cv::mean(reproj_error);

		std::cout<<"[TriangulatePoints] meanErr =  "<<me[0]<<std::endl;

		return me[0];
	}

	bool CMultiCameraPnP::TriangulatePointsBetweenViews(int older_view,int working_view,camera_params_t& camera1, camera_params_t& camera2,std::vector<ImageKeyVector>& pt_view )
	{
		std::vector<PointData> new_triangulated;

		//adding more triangulated points to general cloud
		double reproj_error = TriangulatePoints(older_view,working_view,camera1,camera2,new_triangulated);

		//#ifdef _DEBUG
		std::cout << "[TriangulatePointsBetweenViews] triangulation reproj error " << reproj_error<<" /"<<new_triangulated.size()<< std::endl;
		//#endif //_DEBUG

		if(reproj_error > 20.0 || reproj_error < 1e-2 ) 
		{
			// somethign went awry, delete those triangulated points
			std::cerr << "[TriangulatePointsBetweenViews] reprojection error."<<std::endl;

			return false;
		}

		std::vector<PointData>::iterator p3d_iter =  new_triangulated.begin();

		std::ofstream fout("init_pair.txt");

		int pt3d_num = 0;
		for(;p3d_iter != new_triangulated.end(); ++p3d_iter )
		{
			//add new cloud points
			std::vector<view_t> views = p3d_iter->m_views;
			ImageKeyVector new_views;

			for (int j = 0; j < views.size(); j++)
			{
				int ith_camera = views[j].image;
				int m_idx =  views[j].key;

				GetImageKey(ith_camera,m_idx).m_extra = m_point_data_index;

				int track_idx = GetImageKey(ith_camera,m_idx).m_track;
				m_matchTracks.m_track_data[track_idx].m_extra = m_point_data_index;

				new_views.push_back(ImageKey(j, m_idx));

			}

			pt_view.push_back(new_views);

			m_point_data_index++;
			fout <<p3d_iter->m_pos[0]<<" "<<p3d_iter->m_pos[1]<<" "<<p3d_iter->m_pos[2]<< std::endl;	

			m_point_data.push_back(*p3d_iter);
			pt3d_num++;

		}

		fout.close();
		std::cout<<"[TriangulatePointsBetweenViews] new pt3d num: "<<pt3d_num <<" totle pt3d num: "<<m_point_data_index<<std::endl;

		return true;
	}

	bool CMultiCameraPnP::match()
	{
		// calculate siftgpu feature and matches
		if(!read_sift_features())
		{
			std::cerr<<"read sift feature and match err."<<std::endl;
			return false;
		}

		// compute tracks
		m_matchTracks.InitMatchTable(prune_matches_matrix,prune_images_points,prune_F_matrix,m_image_masks,n_images);

		m_matchTracks.ComputeGeometricConstraints();

		/* Set track pointers to -1 */
		for (int i = 0; i < (int) m_matchTracks.m_track_data.size(); i++) {
			m_matchTracks.m_track_data[i].m_extra = -1;
		}

		return true;
	}

	bool sort_by_first(std::pair<int, std::pair<int, int> > a, std::pair<int, std::pair<int, int> > b) { return a.first > b.first; }
	bool sort_by_second(std::pair<MatchIndex,int> a,std::pair<MatchIndex,int> b) { return a.second > b.second ;}
	
	// Get Image Keypoint with m_image_data.m_keys
	std::vector<cv::Point2f> CMultiCameraPnP::GetImageKeypoints(int image)
	{
		std::vector<cv::Point2f> points;

		ImageData& img_data = m_matchTracks.m_image_data[image];

		for (int i = 0 ; i < img_data.m_keys.size(); ++i)
		{
			cv::Point2f p = cv::Point2f(img_data.m_keys[i].m_x,img_data.m_keys[i].m_y);
			points.push_back(p);
		}

		return points;
	}

	Keypoint& CMultiCameraPnP::GetImageKey(int img, int key)
	{
		return m_matchTracks.m_image_data[img].m_keys[key];
	}
	//Set up the initial cameras
	int CMultiCameraPnP::SetupInitialCameraPair(int i_best, int j_best, std::vector<ImageKeyVector>& pt_view)
	{
		assert(i_best != j_best);

		// set matches between i_best and j_best
		m_matchTracks.SetMatchesFromTracks(i_best, j_best);

		m_matchTracks.SetTracks(i_best);
		m_matchTracks.SetTracks(j_best);

		// Calculates a fundamental matrix
		std::vector<cv::Point2f> src_pt1, src_pt2;

		src_pt1 = GetImageKeypoints(i_best);
		src_pt2 = GetImageKeypoints(j_best);

		MatchIndex idx = GetMatchIndex(i_best,j_best);
		std::vector<KeypointMatch> &list = m_matchTracks.m_matches_table.GetMatchList(idx);
		
		int match_num = list.size();
		for (int i = 0; i < src_pt1.size(); i++)
		{
			src_pt1[i].x -= m_cx;
			src_pt1[i].y -= m_cy;
		}

		for (int i = 0; i < src_pt2.size(); i++)
		{
			src_pt2[i].x -= m_cx;
			src_pt2[i].y -= m_cy;
		}

#ifdef USE_BUNDLER_EPOSE
		int (*_buf)[2] = new int[match_num][2];
		for (int i = 0; i < match_num; i++)
		{
			_buf[i][0] = list[i].m_idx1;
			_buf[i][1] = list[i].m_idx2;
		}
		// Estimate the seconde camera rotation and translation
		bool success = EstimatePose(src_pt1,src_pt2,match_num,_buf,
			m_cameras[0],m_cameras[1]);
	
		if(!success)
			return false;

		std::vector<cv::Point3f> pt3d;
		std::vector<int> indexes;

		int point_num = Triangulate(src_pt1,src_pt2,match_num,_buf,m_cameras[0],m_cameras[1],pt3d,indexes);
		
		std::cout<< point_num <<std::endl;

		// indexes 保存计算三维点云对应的匹配点编号
		for (int i =0; i < indexes.size(); i++)
		{
			int idx = indexes[i];
			int key_idx1 = list[idx].m_idx1;
			int key_idx2 = list[idx].m_idx2;

			//////////////////////////////////////////////////////////////////////////
			// set the index of cloud point
			GetImageKey(i_best,key_idx1).m_extra = i;
			GetImageKey(j_best,key_idx2).m_extra = i;

			int track_idx = GetImageKey(i_best,key_idx1).m_track;
			m_matchTracks.m_track_data[track_idx].m_extra = i;
			//////////////////////////////////////////////////////////////////////////
			// add view
			ImageKeyVector views;
			views.push_back(ImageKey(0, key_idx1));
			views.push_back(ImageKey(1, key_idx2));
			pt_view.push_back(views);

			//////////////////////////////////////////////////////////////////////////
			//add 3d points
			cv::Point3f point = pt3d[i];
			PointData cp;

			cp.m_pos[0] = point.x;
			cp.m_pos[1] = point.y;
			cp.m_pos[2] = point.z;

			view_t v1,v2;
			v1.image = i_best; v1.key = key_idx1;
			v1.x = GetImageKey(i_best,key_idx1).m_x - m_cx;
			v1.y = GetImageKey(i_best,key_idx1).m_y - m_cy;

			v2.image = j_best;  v2.key = key_idx2;
			v2.x = GetImageKey(j_best,key_idx2).m_x - m_cx;
			v2.y = GetImageKey(j_best,key_idx2).m_y - m_cy;

			cp.m_views.push_back(v1);
			cp.m_views.push_back(v2);

			cp.m_ref_image = i_best;

			cp.m_color[0] = (int)GetImageKey(i_best,key_idx1).m_b;
			cp.m_color[1] = (int)GetImageKey(i_best,key_idx1).m_g;
			cp.m_color[2] = (int)GetImageKey(i_best,key_idx1).m_r;

			m_point_data.push_back(cp);
			m_point_data_index++;
		}

		delete[] _buf;
#else
		std::vector<cv::Point2f> pt1,pt2;
		for (int i = 0; i < match_num; i++)
		{
			cv::Point2f p1 = src_pt1[list[i].m_idx1];
			cv::Point2f p2 = src_pt2[list[i].m_idx2];

			pt1.push_back(p1);
			pt2.push_back(p2);
		}

		cv::Mat F = cv::findFundamentalMat(pt1,pt2,CV_FM_RANSAC,0.1,0.99);

		cv::Mat F_64f;
		F.convertTo(F_64f,CV_64F);
		//Essential matrix: compute then extract cameras [R|t]
		if(!FindCameraMatrices(F_64f,m_cameras[0],m_cameras[1],pt1[0],pt2[0]))
			return 0;

		bool good_triangulation = TriangulatePointsBetweenViews(i_best,j_best,m_cameras[0],m_cameras[1],pt_view);

#endif // USE_BUNDLER_EPOSE
		
		m_matchTracks.m_matches_table.ClearMatch(idx);

		return m_point_data_index;
	}

	/* Pick a good initial pair of cameras to bootstrap the bundle
	 * adjustment */
	void CMultiCameraPnP::PickInitialPair(int &i_best, int &j_best)
	{
		/* Compute the match matrix */
		int num_images = n_images;
		int max_matches = 0;
		double max_score = 0.0;
		int max_matches_2 = 0;
		double max_score_2 = 0.0;
		i_best = j_best = -1;
		int i_best_2 = -1, j_best_2 = -1;
		int i_best_3 = -1;
		int j_best_3 = -1;  
		double SCORE_THRESHOLD = 2.0;

		if (m_initial_pair[0] != -1 && m_initial_pair[1] != -1) {
			i_best = m_initial_pair[0];
			j_best = m_initial_pair[1];

			printf("[PickInitialPair] Setting initial pair to "
				"%d and %d\n", i_best, j_best);

			return;
		}

		HashMapTranInfo m_transforms = m_matchTracks.m_transforms;
		/* Compute score for each image pair */
		int max_pts = 0;
		for (int i = 0; i < num_images; i++) {       
			for (int j = i+1; j < num_images; j++) {          
				MatchIndex idx = GetMatchIndex(i, j);
				int num_matches = GetNumTrackMatches(i, j);
				max_pts += num_matches;
#define MATCH_THRESHOLD 50
#define MIN_SCORE 1.0e-1
#define MIN_MATCHES 80
				if (num_matches <= MATCH_THRESHOLD) continue;

				double score = 0.0;
				double ratio = m_transforms[idx].m_inlier_ratio;
				if (ratio == 0.0) score = MIN_SCORE;
				else  score = 1.0 / m_transforms[idx].m_inlier_ratio;
				/* Compute the primary score */
				if (num_matches > max_matches && score > SCORE_THRESHOLD)
				{
					max_matches = num_matches;
					max_score = score;
					i_best = i;
					j_best = j;
				}
				/* Compute the backup score */
				if (num_matches > MIN_MATCHES && score > max_score_2) {
					max_matches_2 = num_matches;
					max_score_2 = score;
					i_best_2 = i;
					j_best_2 = j;
				}
         } // j	
       } //i

		/* Set track pointers to -1 (GetNumTrackMatches alters these
			* values) */
		for (int i = 0; i < (int) m_matchTracks.m_track_data.size(); i++)
			m_matchTracks.m_track_data[i].m_extra = -1;

		if (i_best == -1 && j_best == -1) 
		{
			if (i_best_2 == -1 && j_best_2 == -1) 
			{
				printf("[PickInitialPair] Error: no good camera pairs\n ,Picking first two cameras...\n");
				i_best = 0;
				j_best = 1;
			}
		}
		else
		{
			i_best = i_best_2;
			j_best = j_best_2;
		}
	
		printf("[PickInitialPair] initial pair of cameras: %d, %d\n",i_best,j_best);
	}

	//通过两幅图中的m_visible_points的交集，计算匹配特征点个数
	int CMultiCameraPnP::GetNumTrackMatches(int img1, int img2) 
	{
		std::vector<ImageData> m_image_data = m_matchTracks.m_image_data;
		std::vector<TrackData> m_track_data = m_matchTracks.m_track_data;

		const std::vector<int> &tracks1 = m_image_data[img1].m_visible_points;
		const std::vector<int> &tracks2 = m_image_data[img2].m_visible_points;

		// std::vector<int> isect = GetVectorIntersection(tracks1, tracks2);
		// int num_isect = (int) isect.size();

		std::vector<int>::const_iterator iter;
		for (iter = tracks2.begin(); iter != tracks2.end(); iter++) {
			int track_idx = *iter;
			m_track_data[track_idx].m_extra = 0;
		}

		for (iter = tracks1.begin(); iter != tracks1.end(); iter++) {
			int track_idx = *iter;
			m_track_data[track_idx].m_extra = 1;
		}

		int num_isect = 0;
		for (iter = tracks2.begin(); iter != tracks2.end(); iter++) {
			int track_idx = *iter;
			num_isect += m_track_data[track_idx].m_extra;
		}

		return num_isect;
	}

	bool CMultiCameraPnP::GetBaseLineTriangulation(std::vector<ImageKeyVector>& point_views)
	{
#ifdef _DEBUG
		std::cout << "=========================== Baseline triangulation ===========================\n";
#endif // _DEBUG

		PickInitialPair(m_first_view,m_second_view);

		if(m_first_view == m_second_view)
			return false;

// 		cv::Mat drawImage = drawImageMatches(m_first_view,m_second_view);
// 		cv::imshow("image_match",drawImage);
// 		cv::waitKey(10);

		//////////////////////////////////////////////////////////////////////////
		//add camera by sort
		InitializeCameraParams(m_cameras[0]);
		InitializeCameraParams(m_cameras[1]);

		curr_num_cameras = 2;
		/* Put first camera at origin */
		m_cameras[0].R[0] = 1.0;  m_cameras[0].R[1] = 0.0;  m_cameras[0].R[2] = 0.0;
		m_cameras[0].R[3] = 0.0;  m_cameras[0].R[4] = 1.0;  m_cameras[0].R[5] = 0.0;
		m_cameras[0].R[6] = 0.0;  m_cameras[0].R[7] = 0.0;  m_cameras[0].R[8] = 1.0;

		m_cameras[0].t[0] = 0.0;
		m_cameras[0].t[1] = 0.0;
		m_cameras[0].t[2] = 0.0;

		m_cameras[0].f = m_cameras[1].f = m_f;
		//////////////////////////////////////////////////////////////////////////

		int pt3d_count = SetupInitialCameraPair(m_first_view,m_second_view,point_views);
		
		std::cout<< "[GetBaseLineTriangulation] Initial 3d points count:  "<< pt3d_count<<std::endl;

		return pt3d_count > 0 ;
	}

	bool DecomposeEtoRandT(	cv::Mat_<double>& E,cv::Mat_<double>& R1,cv::Mat_<double>& R2,
		cv::Mat_<double>& t1,cv::Mat_<double>& t2) 
	{
		//Using HZ E decomposition
		cv::SVD svd(E, cv::SVD::MODIFY_A);
		cv::Mat svd_u = svd.u;
		cv::Mat svd_vt = svd.vt;
		cv::Mat svd_w = svd.w;
		
//		std::cout<<"[DecomposeE] "<<svd_w<<std::endl;
		//check if first and second singular values are the same (as they should be)
		double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	
		if( singular_values_ratio > 1.0 )
			singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
		
		if (singular_values_ratio < 0.7) {
			std::cout << "singular values are too far apart."<<std::endl;
			return false;
		}

		cv::Matx33d W(0, -1, 0,//HZ 9.13
			1, 0, 0,
			0, 0, 1);

		cv::Matx33d Wt(0,1,0,
			-1,0,0,
			0,0,1);

		R1 = svd_u * cv::Mat(W) * svd_vt;  //HZ 9.19
		R2 = svd_u * cv::Mat(Wt) * svd_vt; //HZ 9.19
		t1 = svd_u.col(2);  //u3
		t2 = -svd_u.col(2); //u3

		return true;
	}

	/* Check cheirality for a camera and a point */
	bool CheckCheirality(cv::Mat_<double> p, cv::Matx34d P1)
	{
		cv::Mat_<double> X = p;
		cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0), P1(0,1), P1(0,2), 
			P1(1,0), P1(1,1), P1(1,2), 
			P1(2,0), P1(2,1), P1(2,2));

		cv::Mat_<double> cam = R*X;

		cam(0) += P1(0,3);
		cam(1) += P1(1,3);
		cam(2) += P1(2,3);

		if(cam(2) < 0.0 && p(2) < 0)
			return true;
		else
			return false;
	}

	int CMultiCameraPnP::SelectPMatrix(std::vector<cv::Matx34d> _4Pmatrixs,cv::Point3d u,cv::Point3d u1)
	{
		int N = _4Pmatrixs.size();

		cv::Matx34d P(1,0,0,0,
					  0,1,0,0,
					  0,0,1,0);
		int idx = -1;
		int maxZ = 0.0;
		for (int i = 0; i < N ; i++)
		{
			cv::Matx34d P1 = _4Pmatrixs[i];

			cv::Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1); //(4X1)
			cv::Mat_<double> pt3d = (cv::Mat_<double>(3,1) << X(0),X(1),X(2));

			bool front = CheckCheirality(pt3d,P);
			bool front1 = CheckCheirality(pt3d,P1);
			if(front && front1)
			{
				idx = i;
				break;
			}
		}
		
		
		return idx;
	}

	int CMultiCameraPnP::SelectPMatrix(std::vector<cv::Matx34d> _4Pmatrixs,cv::Point2f u,cv::Point2f u1)
	{

		int N = _4Pmatrixs.size();
		int idx = -1;

		for (int i = 0; i < N ; i++)
		{
			cv::Matx34d P1 = _4Pmatrixs[i];

			m_cameras[1].R[0] = P1(0,0);  m_cameras[1].R[1] = P1(0,1);  m_cameras[1].R[2] = P1(0,2);
			m_cameras[1].R[3] = P1(1,0);  m_cameras[1].R[4] = P1(1,1);  m_cameras[1].R[5] = P1(1,2);
			m_cameras[1].R[6] = P1(2,0);  m_cameras[1].R[7] = P1(2,1);  m_cameras[1].R[8] = P1(2,2);

			m_cameras[1].t[0] = P1(0,3);
			m_cameras[1].t[1] = P1(1,3);
			m_cameras[1].t[2] = P1(2,3);

			cv::Point3f point3d;
			bool in_front = Triangulate(u,u1,m_cameras[0],m_cameras[1],point3d);

			if(in_front)
			{
				idx = i;
				break;
			}
		}

		return idx;
	}

	bool CMultiCameraPnP::FindCameraMatrices(cv::Mat F,cv::Matx34d& P, cv::Matx34d& P1 , cv::Point2f kp,cv::Point2f kp1 )
	{
		if(F.empty())
			return false;
		//Essential matrix: compute then extract cameras [R|t]
		camera_params_t cam1 = m_cameras[0];
		camera_params_t cam2 = m_cameras[1];

		double K[9],K1[9];
		GetIntrinsics(cam1,K);
		GetIntrinsics(cam2,K1);

		cv::Mat mK = cv::Mat(3,3,CV_64F,K);
		cv::Mat mK1 = cv::Mat(3,3,CV_64F,K1);
		cv::Mat_<double> E = mK1.t() * F * mK; //according to HZ (9.12)

		double detE = fabs(cv::determinant(E));
		if ( detE > 1e-06) 
		{
			std::cout << "[FindCameraMatrices] det(E) != 0 : " << detE<< std::endl;
			P1 = 0;
			return false;
		}

		//decompose E to P' , HZ (9.19)
		cv::Mat_<double> R1(3,3);
		cv::Mat_<double> R2(3,3);
		cv::Mat_<double> t1(1,3);
		cv::Mat_<double> t2(1,3);

		if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

		double detR1 = (cv::determinant(R1));
		double detR2 = (cv::determinant(R2));
	
		if(fabs(detR1 + 1.0) < 1e-06 || fabs(detR2 + 1.0) < 1e-06)
		{
			E = -E;
			if (!DecomposeEtoRandT(E,R1,R2,t1,t2))
				return false;
		}

		detR1 = (cv::determinant(R1));
		detR2 = (cv::determinant(R2));


		std::vector<cv::Matx34d> _4Pmatrixs;
		if ( fabs( fabs(detR1)- 1.0) < 1e-06)
		{
			cv::Matx34d _p1 = cv::Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0),	R1(2,1),	R1(2,2),	t1(2));

			cv::Matx34d _p2 = cv::Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	-t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	-t1(1),
				R1(2,0),	R1(2,1),	R1(2,2),	-t1(2));

			_4Pmatrixs.push_back(_p1);
			_4Pmatrixs.push_back(_p2);
		}

		if(fabs( fabs(detR2)- 1.0) < 1e-06)
		{
			cv::Matx34d _p3 = cv::Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
				R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
				R2(2,0),	R2(2,1),	R2(2,2),	t2(2));

			cv::Matx34d _p4 = cv::Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	-t2(0),
				R2(1,0),	R2(1,1),	R2(1,2),	-t2(1),
				R2(2,0),	R2(2,1),	R2(2,2),	-t2(2));

			_4Pmatrixs.push_back(_p3);
			_4Pmatrixs.push_back(_p4);
		}


		if(_4Pmatrixs.empty())
		{
			std::cerr << "[FindCameraMatrices] |det(R)| != 1.0 " << std::endl;
			P1 = 0;
			return false;
		}

		int i = SelectPMatrix(_4Pmatrixs,kp,kp1);
		if(i < 0)
			return false;
		else
			P1 = _4Pmatrixs[i];


		//////////////////////////////////////////////////////////////////////////
		//camera center
		cv::Mat_<double> R = i > 1 ? R2: R1;
		cv::Mat_<double> t = i > 1 ? (i == 0 ? t1 : -t1 ):( i == 2? t2 : -t2);

		std::cout<< R <<std::endl<<t.t()<<std::endl;
		std::cout<< " Camera Center: "<<-R*t  <<std::endl;
		return true;
	}
	// Find P matrix
	bool CMultiCameraPnP::FindCameraMatrices(cv::Mat F,camera_params_t& camera1, camera_params_t& camera2,cv::Point2f kp1,cv::Point2f kp2)
	{

		double K1[9],K2[9];
		GetIntrinsics(camera1,K1);
		GetIntrinsics(camera2,K2);
		
		//Essential matrix: compute then extract cameras [R|t]
		cv::Mat mK1 = cv::Mat(3,3,CV_64F,K1);
		cv::Mat mK2 = cv::Mat(3,3,CV_64F,K2);

		std::cout<<mK1<<std::endl;

		cv::Mat_<double> E = mK2.t() * F * mK1; //according to HZ (9.12)

		E(0,0) = - E(0,0);
		E(0,1) = - E(0,1);
		E(1,0) = - E(1,0);
		E(1,1) = - E(1,1);
		E(2,2) = - E(2,2);

		double detE = fabs(cv::determinant(E));

		if ( detE > 1e-06) 
		{
			std::cout << "[FindCameraMatrices] det(E) != 0 : " << detE<< std::endl;
			return false;
		}

		//decompose E to P' , HZ (9.19)
		cv::Mat_<double> R1(3,3);
		cv::Mat_<double> R2(3,3);
		cv::Mat_<double> t1(1,3);
		cv::Mat_<double> t2(1,3);

		if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

		double detR1 = (cv::determinant(R1));
		double detR2 = (cv::determinant(R2));
		
		if(fabs(detR1 + 1.0) < 1e-06 || fabs(detR2 + 1.0) < 1e-06)
		{
			E = -E;
			if (!DecomposeEtoRandT(E,R1,R2,t1,t2))
				return false;

			detR1 = (cv::determinant(R1));
			detR2 = (cv::determinant(R2));
		}

		std::vector<cv::Matx34d> _4Pmatrixs;
		if ( fabs( fabs(detR1)- 1.0) < 1e-06)
		{
			cv::Matx34d _p1 = cv::Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0),	R1(2,1),	R1(2,2),	t1(2));

			cv::Matx34d _p2 = cv::Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	-t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	-t1(1),
				R1(2,0),	R1(2,1),	R1(2,2),	-t1(2));

			_4Pmatrixs.push_back(_p1);
			_4Pmatrixs.push_back(_p2);
		}
			
		if(fabs( fabs(detR2)- 1.0) < 1e-06)
		{
			cv::Matx34d _p3 = cv::Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
				R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
				R2(2,0),	R2(2,1),	R2(2,2),	t2(2));

			cv::Matx34d _p4 = cv::Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	-t2(0),
				R2(1,0),	R2(1,1),	R2(1,2),	-t2(1),
				R2(2,0),	R2(2,1),	R2(2,2),	-t2(2));

			_4Pmatrixs.push_back(_p3);
			_4Pmatrixs.push_back(_p4);
		}


		if(_4Pmatrixs.empty())
		{
			std::cerr << "[FindCameraMatrices] |det(R)| != 1.0 " << std::endl;
			return false;
		}

		cv::Point3d ku1(kp1.x,kp1.y,-1),ku2(kp2.x,kp2.y,-1);

		cv::Mat_<double> unorm1 = mK1.inv() * cv::Mat_<double>(ku1);
		ku1 = unorm1.at<cv::Point3d>(0);

		cv::Mat_<double> unorm2 = mK2.inv() * cv::Mat_<double>(ku2);
		ku2 = unorm2.at<cv::Point3d>(0);

		int i = SelectPMatrix(_4Pmatrixs,ku1,ku2);

		if(i < 0)
			return false;
		else
		{
			cv::Matx34d P = _4Pmatrixs[i];
			std::cout<<"R,t"<<std::endl;
			std::cout<<setprecision(6);
			std::cout<<P<<std::endl;

			camera2.R[0] = P(0,0),camera2.R[1] = P(0,1),camera2.R[2] = P(0,2);
			camera2.R[3] = P(1,0),camera2.R[4] = P(1,1),camera2.R[5] = P(1,2);
			camera2.R[6] = P(2,0),camera2.R[7] = P(2,1),camera2.R[8] = P(2,2);

			camera2.t[0] = P(0,3), camera2.t[1] = P(1,3),camera2.t[1] = P(2,3);
		}

		return true;
	}

	void CMultiCameraPnP::RecoverDepthFromImages()
	{
		std::cout << "======================================================================\n";
		std::cout << "======================== Depth Recovery Start ========================\n";
		std::cout << "======================================================================\n";
		
		std::vector<ImageKeyVector> point_views;
		
		char ch[256];
		strcpy(ch,m_path);

		if(!GetBaseLineTriangulation(point_views))
			return ;
		
		done_views.push_back(m_first_view);
		done_views.push_back(m_second_view);
		good_views.push_back(m_first_view);
		good_views.push_back(m_second_view);

 		AdjustCurrentBundle(point_views,false);
		char tmp[128];
		sprintf(tmp,"/00_output[%d,%d].txt",m_first_view,m_second_view);
		strcat(ch,tmp);
		WriteCloudPoint(ch,0);

		int order = 0;
		while (done_views.size() < n_images)
		{
			std::cout<<"[LOOP ]cloud size: "<<m_point_data_index<<std::endl;
			//find image with highest 2d-3d correspondance [Snavely07 4.2]
			int max_2d3d_view = -1, max_2d3d_count = 0;

			for (int _i=0; _i < n_images; _i++) 
			{	
				std::vector<int>::iterator it = find(done_views.begin(),done_views.end(),_i);

				if(it != done_views.end()) 
					continue; //already done with this view				
				
				/* Find the tracks seen by this image */
				std::vector<int> & tracks = m_matchTracks.m_image_data[_i].m_visible_points;

				int num_track = tracks.size();
				int used_track_num = 0;
				for(int j = 0 ; j < num_track ; j++)
				{
					int tr = tracks[j];
					// Find the 3d cloud point index
					int pt = m_matchTracks.m_track_data[tr].m_extra;
					if(pt < 0 )
						continue;

					if((int)point_views[pt].size() == 0)
						continue;

					used_track_num++;
				}

				if(max_2d3d_count < used_track_num)
				{
					max_2d3d_count = used_track_num;
					max_2d3d_view = _i;
				}
			}

			std::cout<<"[RecoverDepthFromImages] Matched View: "<<max_2d3d_view<<" Matched Count: "<<max_2d3d_count<<std::endl;
			 //highest 2d3d matching view
			int i = max_2d3d_view;
			done_views.push_back(i);
			
			cout<< done_views.size() << endl;

			if(max_2d3d_view < 0)
				continue;

			cv::Mat_<double> R,t;
			m_matchTracks.SetTracks(i);

			bool pose_estimated = FindPoseEstimation(i,t,R,point_views);
			if(!pose_estimated)
				continue;
			good_views.push_back(i);

			//CheckPointKeyConsistency(point_views,good_views);
			AdjustCurrentBundle(point_views,false);
// 			sprintf(ch,"basecloud/01_output[%d,%d].txt",m_first_view,m_second_view);
// 			WriteCloudPoint(ch,0);

			// have matched 3points, and updated the camera and pt_view
			int cloud_size_befor_triangulation = m_point_data_index;

			// start triangulating with previous GOOD views
			std::vector<int>::iterator good_view = good_views.begin();
			int camera_id = 0;
			for (; good_view != good_views.end(); ++camera_id,++good_view) 
			{
				int startIdx = m_point_data_index;
				int view = *good_view;
				if( view == i )
					continue; //skip current...				
				m_matchTracks.SetMatchesFromTracks(view, i);
				m_matchTracks.SetTracks(view);
				std::vector<cv::Point2f> src_pt1, src_pt2;

				src_pt1 = GetImageKeypoints(view);
				src_pt2 = GetImageKeypoints(i);

				MatchIndex idx = GetMatchIndex(view,i);
				std::vector<KeypointMatch> &list = m_matchTracks.m_matches_table.GetMatchList(idx);
				int match_num = list.size();
				if( match_num < 8 )
				{
					m_matchTracks.m_matches_table.ClearMatch(GetMatchIndex(view, i));
					continue;
				}

				for (int j = 0; j < src_pt1.size(); j++)
				{
					src_pt1[j].x -= m_cx;
					src_pt1[j].y -= m_cy;
				}

				for (int j = 0; j < src_pt2.size(); j++)
				{
					src_pt2[j].x -= m_cx;
					src_pt2[j].y -= m_cy;
				}

				int (*_buf)[2] = new int[match_num][2];
				int new_match_num = 0;
				for (int j = 0; j < match_num; j++)
				{
					int key_idx = list[j].m_idx1;
					int pt_idx = GetImageKey(view,key_idx).m_extra;
					if(pt_idx >= 0) // 当前点对应有3维点
					{
						//add pt_view
						ImageKeyVector& m_view =  point_views[pt_idx];
						bool add_view = false;
						for (int k = 0; k < m_view.size() ; k++)
						{
							if(m_view[k].first == camera_id)  // already exist?
							{
								add_view = true;
								break;
							}
						}
						// the pt_view created after Triangulate, add match view
						if(!add_view)
						{	
							m_view.push_back(ImageKey(camera_id,key_idx));
							view_t v;
							v.image = view; v.key = key_idx;
							v.x = GetImageKey(view,key_idx).m_x - m_cx;
							v.y = GetImageKey(view,key_idx).m_y - m_cy;
							m_point_data[pt_idx].m_views.push_back(v);
						}

						continue;
					}

					_buf[new_match_num][0] = list[j].m_idx1;
					_buf[new_match_num][1] = list[j].m_idx2;

					new_match_num++;
				}

				
				std::cout<<"match size: "<<match_num<<" "<<new_match_num<<std::endl;
				std::cout<<"camera num: "<< curr_num_cameras <<std::endl;
				std::vector<cv::Point3f> pt3d;
				std::vector<int> indexes;

				int point_num = Triangulate(src_pt1,src_pt2,new_match_num,_buf,m_cameras[camera_id],m_cameras[curr_num_cameras -1],pt3d,indexes);
				std::cout<< point_num <<std::endl;

				//add 3d point cloud
				for (int j=0; j < point_num; j++)
				{
					int k = indexes[j];
					int m_idx1 = _buf[k][0];
					int m_idx2 = _buf[k][1];

					GetImageKey(view,m_idx1).m_extra = m_point_data_index;
					GetImageKey(i,   m_idx2).m_extra = m_point_data_index;

					//新添加三维点对应的轨迹track
					int track_idx = GetImageKey(i,m_idx2).m_track;
					m_matchTracks.m_track_data[track_idx].m_extra = m_point_data_index;

					ImageKeyVector pt_view;
					pt_view.push_back(ImageKey(camera_id,m_idx1));
					pt_view.push_back(ImageKey(curr_num_cameras-1,m_idx2));
					point_views.push_back(pt_view);

					cv::Point3f point = pt3d[j];
					PointData cp;

					cp.m_pos[0] = point.x;
					cp.m_pos[1] = point.y;
					cp.m_pos[2] = point.z;

					view_t v1 ,v2;
					v1.image = view; v1.key = m_idx1;
					v1.x = GetImageKey(view,m_idx1).m_x - m_cx;
					v1.y = GetImageKey(view,m_idx1).m_y - m_cy;

					v2.image = i ; v2.key = m_idx2;
					v2.x = GetImageKey(i,m_idx2).m_x - m_cx;
					v2.y = GetImageKey(i,m_idx2).m_y - m_cy;

					cp.m_views.push_back(v1);
					cp.m_views.push_back(v2);

					cp.m_ref_image = i;

					cp.m_color[0] = (int)GetImageKey(i,m_idx2).m_b;
					cp.m_color[1] = (int)GetImageKey(i,m_idx2).m_g;
					cp.m_color[2] = (int)GetImageKey(i,m_idx2).m_r;

					m_point_data.push_back(cp);
					m_point_data_index++;
				}

//				AdjustCurrentBundle(point_views);
// 				char ch[64];
// 				sprintf(ch,"basecloud/%02d_output[%d].txt",i,view);
// 				WriteCloudPoint(ch,startIdx);

				m_matchTracks.m_matches_table.ClearMatch(GetMatchIndex(view, i));
				delete []_buf;
			}

			int cloud_size_after_triangulation = m_point_data_index;
			int added_point_num = (cloud_size_after_triangulation - cloud_size_befor_triangulation);
			std::cout<<"[RecoverDepthFromImages] Added cloud points: " <<added_point_num<<std::endl;
			
			//CheckPointKeyConsistency(point_views,good_views);
			AdjustCurrentBundle(point_views,false);

			if(added_point_num == 0)
				continue;
	
			char tmp[64];
			sprintf(tmp,"/%02d_%02d_output.txt",++order,i);
			strcpy(ch,m_path);
			strcat(ch,tmp);
			WriteCloudPoint(ch,cloud_size_befor_triangulation);
			
		}
		
		//CheckPointKeyConsistency(point_views,good_views);
		//AdjustCurrentBundle(point_views,true);

		sprintf(tmp,"/full_output.txt");
		strcpy(ch,m_path);
		strcat(ch,tmp);
		WriteCloudPoint(ch,0);

		Bundler2PMVS(point_views);
		
		sprintf(tmp,"/model.nvm");
		strcpy(ch,m_path);
		strcat(ch,tmp);
		SaveModelFile(ch,point_views);

		

		std::cout << "======================================================================\n";
		std::cout << "========================= Depth Recovery DONE ========================\n";
		std::cout << "======================================================================\n";
	}

	bool RefineCamera(camera_params_t& camera,std::vector<cv::Point3f>& points3d,std::vector<cv::Point2f> points2d)
	{
		// bundler adjustment
		//////////////////////////////////////////////////////////////////////////
		int num_inliers = points3d.size();

		std::vector<CameraT> camera_data;
		std::vector<Point3D> point_data;
		std::vector<Point2D> measurements;
		std::vector<int> ptidx;
		std::vector<int> camidx;

		CameraT _cam;
		double t[3], d[2];			
		double focla_length = camera.f;
		_cam.SetFocalLength(focla_length);
		// R,T
		_cam.SetMatrixRotation(camera.R);
		
		t[0] = -(camera.R[0]*camera.t[0] + camera.R[1]*camera.t[1] + camera.R[2]*camera.t[2]);
		t[1] = -(camera.R[3]*camera.t[0] + camera.R[4]*camera.t[1] + camera.R[5]*camera.t[2]);
		t[2] = -(camera.R[6]*camera.t[0] + camera.R[7]*camera.t[1] + camera.R[8]*camera.t[2]);
		_cam.SetTranslation(t);
		//distortion
		d[0] = d[1] = 0;
		_cam.SetNormalizedMeasurementDistortion(d[0]);
		camera_data.push_back(_cam);

		for (int i = 0; i < num_inliers; i++)
		{
			Point3D pt;
			pt.SetPoint(points3d[i].x,points3d[i].y,points3d[i].z);
			point_data.push_back(pt);

			Point2D mp;
			mp.SetPoint2D(-points2d[i].x,-points2d[i].y);
			measurements.push_back(mp);

			camidx.push_back(0);
			ptidx.push_back(i);
		}


		BundleAdjuster mba;
		mba.RunBundleAdjustment(camera_data,point_data,measurements,ptidx,camidx);

		_cam = camera_data[0];
		camera.f = _cam.GetFocalLength();

		_cam.GetMatrixRotation(camera.R);
		_cam.GetCameraCenter(camera.t);

		points3d.clear();
		for (int i=0; i < num_inliers; i++)
		{
			double pt[3];
			point_data[i].GetPoint(pt);
			points3d.push_back(cv::Point3f(pt[0],pt[1],pt[2]));

		}
		//////////////////////////////////////////////////////////////////////////

		return true;
	}

	std::vector<int> CMultiCameraPnP::ProjectMeanError(camera_params_t camera,std::vector<cv::Point3f> points3d,std::vector<cv::Point2f> points2d)
	{
		std::vector<int> inliers;
		double f = camera.f;
		cv::Mat_<double> K = (cv::Mat_<double>(3,3)<< f, 0 , 0,
													  0 ,f, 0,
													  0 , 0 , 1);
		cv::Mat_<double> R = (cv::Mat_<double>(3,3)<<camera.R[0], camera.R[1], camera.R[2],
													 camera.R[3], camera.R[4], camera.R[5],
													 camera.R[6], camera.R[7], camera.R[8]);

		cv::Mat_<double> C = (cv::Mat_<double>(3,1)<<camera.t[0],camera.t[1],camera.t[2]);

		cv::Mat_<double> T = -R*C;

		std::vector<cv::Point2f> proj_pts2d;
		projectPoints(points3d,R,T,K,cv::Mat(),proj_pts2d);

		float sum_reprj_err = 0.0;
		float max_err = 0;
		int j = 0;
		for(int i = 0 ;i< points3d.size() ;i++) {
			
			if(points2d[i].x*proj_pts2d[i].x < 0) // 异号
			{
				points2d[i].x = - points2d[i].x;
				points2d[i].y = - points2d[i].y;
			}

			double re_projErr = cv::norm(proj_pts2d[i]- points2d[i]);

			if(max_err < re_projErr) max_err = re_projErr;

			if(re_projErr > 4.0 )
				continue;

			inliers.push_back(i);

			sum_reprj_err += re_projErr;
			j++;
		}
		if(j > 0) sum_reprj_err /= j ;

		printf("[ProjectMeanError] num = %d ,re_projection error = %5.3f,max error = %5.3f\n ",j,sum_reprj_err,max_err);

		return inliers;
	}

	bool CMultiCameraPnP::FindPoseEstimation(int working_view,
				cv::Mat_<double>& t,cv::Mat_<double>& R, std::vector<ImageKeyVector>& pt_view)
	{

		std::vector<cv::Point3f> ppcloud;
		std::vector<cv::Point2f> imgpoints;

		std::vector<int> pt3d_index,pt2d_index;
		
		Find2D3DCorrespondences(working_view,ppcloud,imgpoints,pt_view,pt2d_index,pt3d_index);

		//std::cout<<"[FindPoseEstimation]: 3d: "<<ppcloud.size()<<" 2d: "<<imgpoints.size()<<std::endl;

		if(ppcloud.size() <= 7 || imgpoints.size() <= 7 || ppcloud.size() != imgpoints.size())
		{ 
			//something went wrong aligning 3D to 2D points..
			std::cerr << "[FindPoseEstimation] Couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<std::endl;
			return false;
		}
		
		cv::Mat Ks,Rs,Ts;
		int num_pts = ppcloud.size();
		std::vector<int> inliers,weakinliers,outinliers;
		
		bool is_good = FindAndVerifyCamera(num_pts,ppcloud,imgpoints,pt2d_index,Ks,Rs,Ts,5.0,16*5.0,inliers,weakinliers,outinliers);
		
		if(!is_good )
			return false;

// 		std::cout<<std::endl;
// 		std::cout<<"Solved result: "<<std::endl;
// 		std::cout<<"K: "<<Ks<<std::endl;
// 		std::cout<<"R: "<<Rs<<std::endl;
// 		std::cout<<"T: "<<Ts<<std::endl;
//		cv::Mat_<double> K = (cv::Mat_<double>(3,3)<<m_f, 0 , 0,
//													  0 ,m_f, 0,
//													  0 , 0 , 1);
// 		R = K.inv()*Ks*Rs;
// 		t = K.inv()*Ks*Ts.t();	
// 		for (int i = 0; i < num_pts; i++)
// 		{
// 			cv::Mat_<double> pt3d = (cv::Mat_<double>(3,1)<<ppcloud[i].x,ppcloud[i].y,ppcloud[i].z);
// 			cv::Mat_<double> proj_pt3d = Rs*pt3d + Ts.t();
// 			cv::Mat_<double> proj_pt2d = Ks*proj_pt3d;
// 			std::cout<<proj_pt2d(0)<<" "<<proj_pt2d(1)<<" "<<proj_pt2d(2)<<std::endl;
// 			proj_pts2d.push_back(cv::Point2f(proj_pt2d(0)/proj_pt2d(2),proj_pt2d(1)/proj_pt2d(2)));
// 		}

		if(cv::norm(Ts) > 100.0) 
		{
		// this is bad...
			std::cerr << "[FindPoseEstimation] Estimated camera movement is too big, skip this camera\r\n";
			return false;
		}

		float detR = fabs(determinant(Rs));
		if(fabs(detR - 1.0) > 5e-07)
		{
			std::cerr << "[FindPoseEstimation] Rotation is incoherent. we should try a different base view..." << std::endl;
			return false;
		}
	
		R = Rs;
		t = Ts;

		//set up camera
		InitializeCameraParams(m_cameras[curr_num_cameras]);
		m_cameras[curr_num_cameras].f = 0.5*(Ks.at<double>(0,0) + Ks.at<double>(1,1));

		m_cameras[curr_num_cameras].R[0] = R(0,0),m_cameras[curr_num_cameras].R[1] = R(0,1),m_cameras[curr_num_cameras].R[2] = R(0,2);
		m_cameras[curr_num_cameras].R[3] = R(1,0),m_cameras[curr_num_cameras].R[4] = R(1,1),m_cameras[curr_num_cameras].R[5] = R(1,2);
		m_cameras[curr_num_cameras].R[6] = R(2,0),m_cameras[curr_num_cameras].R[7] = R(2,1),m_cameras[curr_num_cameras].R[8] = R(2,2);
		// -R^T*t
		m_cameras[curr_num_cameras].t[0] = -(R(0,0)*t(0) + R(1,0)*t(1) + R(2,0)*t(2));
		m_cameras[curr_num_cameras].t[1] = -(R(0,1)*t(0) + R(1,1)*t(1) + R(2,1)*t(2));
		m_cameras[curr_num_cameras].t[2] = -(R(0,2)*t(0) + R(1,2)*t(1) + R(2,2)*t(2));

		int num_inliers = weakinliers.size();
		std::vector<cv::Point3f> ppcloud_final;
		std::vector<cv::Point2f> imgpoints_final;
		std::vector<int> pt3d_index_final,pt2d_index_final;
		for (int i = 0; i < num_inliers; i++)
		{
			int idx = weakinliers[i];
			ppcloud_final.push_back(ppcloud[idx]);
			imgpoints_final.push_back(imgpoints[idx]);

			pt3d_index_final.push_back(pt3d_index[idx]);
			pt2d_index_final.push_back(pt2d_index[idx]);
		}

		RefineCamera(m_cameras[curr_num_cameras],ppcloud_final,imgpoints_final);
		std::vector<int> inliers_final = ProjectMeanError(m_cameras[curr_num_cameras],ppcloud_final,imgpoints_final);

		if(inliers_final.size() < 8 )
		{
			std::cerr << "[FindPoseEstimation] Couldn't find [enough] reliable corresponding （Only "<<inliers_final.size() << ")"<<std::endl;
			return false;
		}
		//////////////////////////////////////////////////////////////////////////
		
		//updata m_image_data
		ImageData& img_data = m_matchTracks.m_image_data[working_view];
		
		for (int i=0; i < inliers_final.size(); i++)
		{
			int idx1 = inliers_final[i];
			int pt_idx = pt3d_index_final[idx1];
			int key_idx = pt2d_index_final[idx1];
			//2d image point  correspond to 3d object point idx
			img_data.m_keys[key_idx].m_extra = pt_idx;
			
			pt_view[pt_idx].push_back(ImageKey(curr_num_cameras, key_idx));
			view_t v;
			v.image = working_view; v.key = key_idx;
			v.x = GetImageKey(working_view,key_idx).m_x - m_cx;
			v.y = GetImageKey(working_view,key_idx).m_y - m_cy;

			m_point_data[pt_idx].m_views.push_back(v);

			//3d object point correspond to 2d image view and the image point
			// Test whether have the same track
// 			int m_idx1 = pt_view[pt3d_index[inlier_idx]][0].second;
// 			int m_idx2 = pt_view[pt3d_index[inlier_idx]][1].second;
// 			int tr1 = GetImageKey(m_first_view,m_idx1).m_track;
// 			int tr2 = GetImageKey(m_second_view,m_idx2).m_track;
// 			int tr3 = GetImageKey(working_view,pt2d_index[inlier_idx]).m_track; 
// 			std::cout<<tr1<<" "<<tr2<<" "<< tr3<<std::endl;  
		}

		curr_num_cameras++;

		return true;
	}

	void CMultiCameraPnP::Find2D3DCorrespondences(int working_view, 
	std::vector<cv::Point3f>& ppcloud, 
	std::vector<cv::Point2f>& imgPoints,
	std::vector<ImageKeyVector> pt_view,
	std::vector<int> & keys_solve,
	std::vector<int> & idxs_solve)
	{
		// 需要通过已知的2d->3d 的关系以及 2d->2d 的关系，
		// 获得新的 2d->3d 的关系
		ppcloud.clear();
		imgPoints.clear();

		keys_solve.clear(); // 2d特征点对应的编号
		idxs_solve.clear(); // 3d点云对应的编号

		//working_view = m_second_view;

		//获取在当前帧中，点云匹配的二维点
		ImageData working_imagedata = m_matchTracks.m_image_data[working_view];		
		std::vector<TrackData> track_data =  m_matchTracks.m_track_data;
		std::vector<Keypoint> keypoints = working_imagedata.m_keys;

		int track_num = working_imagedata.m_visible_points.size();

		for(int i =0; i< track_num; i++)
		{
			int tr = working_imagedata.m_visible_points[i];
			if(track_data[tr].m_extra < 0)
				continue;
			int pt = track_data[tr].m_extra;
			if((int) pt_view[pt].size() == 0)
				continue;

			PointData & pt3d_data = m_point_data[pt];
			cv::Point3f pt3d;
			pt3d.x = pt3d_data.m_pos[0];
			pt3d.y = pt3d_data.m_pos[1];
			pt3d.z = pt3d_data.m_pos[2];
			ppcloud.push_back(pt3d);			

			int pt2d_index = working_imagedata.m_visible_keys[i];
			cv::Point2f pt2d ;
			pt2d.x = (keypoints[pt2d_index].m_x - m_cx);
			pt2d.y = (keypoints[pt2d_index].m_y - m_cy);
					
			imgPoints.push_back(pt2d);
			// add 3d point m_view ?

			keys_solve.push_back(pt2d_index);
			idxs_solve.push_back(pt);
		}
	}

	cv::Mat CMultiCameraPnP::drawImageMatches(int _index_i, int _index_j)
	{
		cv::Mat drawImage = m_matchTracks.drawImageMatches(_index_i,_index_j);

		return drawImage;
	}

	void CMultiCameraPnP::WriteCloudPoint(char filename[],int start_idx)
	{
		std::ofstream fp(filename);

		if(fp.is_open())
		{
			for(int i = start_idx; i < m_point_data_index; i++)
			{
				double *point = m_point_data[i].m_pos;
				int * color = m_point_data[i].m_color;
				fp<< 10*point[0]<<";"<<10*point[1]<<";"<<10*point[2]<<";"<<color[2]<<";"<<color[1]<<";"<<color[0]<<std::endl;
			}
		}
		fp.close();
	}

	void CMultiCameraPnP::SetBundleAdjustData(std::vector<CameraT> &camera_data,std::vector<Point3D>& point_data,
		std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx, std::vector<ImageKeyVector> pt_views)
	{
		int N = curr_num_cameras;
		int M = m_point_data.size();

		std::cout<<"camera num: "<<N<<std::endl;
		int nproj = 0;
		camera_data.resize(N);

		for (int i=0 ; i<N ; i++)
		{
			CameraT _cam;

			double q[9], t[3], d[2];			
			double focla_length = m_cameras[i].f;
			_cam.SetFocalLength(focla_length);
			// R,T
			_cam.SetMatrixRotation(m_cameras[i].R);

			// -R*(-R^T*t) = t
			t[0] = -(m_cameras[i].R[0]*m_cameras[i].t[0] + m_cameras[i].R[1]*m_cameras[i].t[1] + m_cameras[i].R[2]*m_cameras[i].t[2]);
			t[1] = -(m_cameras[i].R[3]*m_cameras[i].t[0] + m_cameras[i].R[4]*m_cameras[i].t[1] + m_cameras[i].R[5]*m_cameras[i].t[2]);
			t[2] = -(m_cameras[i].R[6]*m_cameras[i].t[0] + m_cameras[i].R[7]*m_cameras[i].t[1] + m_cameras[i].R[8]*m_cameras[i].t[2]);

			//std::cout<<i<<" "<<m_cameras[i].t[0]<<" "<<m_cameras[i].t[1]<<" "<<m_cameras[i].t[2]<<std::endl;
			_cam.SetTranslation(t);
			//distortion
			d[0] = m_cameras[i].k[0];
			d[1] = m_cameras[i].k[1];
			_cam.SetNormalizedMeasurementDistortion(d[0]);

			camera_data[i] = _cam;
		}

		int npoint = M;
		point_data.resize(npoint);
		
		for (int i = 0 ; i < npoint; i++)
		{
			float pt[3];

			pt[0] = m_point_data[i].m_pos[0];
			pt[1] = m_point_data[i].m_pos[1];
			pt[2] = m_point_data[i].m_pos[2];

			point_data[i].SetPoint(pt); 

			//2d projection
			ImageKeyVector imkey = pt_views[i];
			for (int k =0; k < imkey.size(); k++) 
			{
				int camera_id = imkey[k].first;
				int im_idx = good_views[camera_id];
				int key_idx = imkey[k].second;

				//std::cout<<im_idx<<" "<<key_idx<<std::endl;

				int view = im_idx, point = i;
				float x = (GetImageKey(im_idx,key_idx).m_x - m_cx );
				float y = (GetImageKey(im_idx,key_idx).m_y - m_cy );

				//std::cout<<imkey[k].first<<" "<<GetImageKey(im_idx,key_idx).m_x <<" "<<GetImageKey(im_idx,key_idx).m_y<<std::endl;

				// the optical center (光学中心)
				measurements.push_back(Point2D(-x,-y));
				camidx.push_back(imkey[k].first);    //camera index
				ptidx.push_back(point);        //point index
				
				nproj ++;
			}		
		}
	}

	void CMultiCameraPnP::GetBundleAdjustData(std::vector<CameraT> &camera_data,std::vector<Point3D>& point_data,
		std::vector<Point2D>& measurements,std::vector<int>& ptidx,std::vector<int>& camidx)
	{
		int N = curr_num_cameras;
		int M = point_data.size();

		int j = 0;
		for(int i = 0; i < N; i++)
		{
			float focal_length = camera_data[i].GetFocalLength();
			//std::cout<<"focal lenght: "<<focal_length<<std::endl;

			m_cameras[i].f = focal_length;

			camera_data[i].GetMatrixRotation(m_cameras[i].R);
			camera_data[i].GetCameraCenter(m_cameras[i].t);//output is the -R^T*t

			m_cameras[i].k[0] = camera_data[i].GetNormalizedMeasurementDistortion();

			//-R^T*t
			//m_cameras[i].t[0] = -(m_cameras[i].R[0]*t[0] + m_cameras[i].R[3]*t[1] + m_cameras[i].R[6]*t[2]);
			//m_cameras[i].t[1] = -(m_cameras[i].R[1]*t[0] + m_cameras[i].R[4]*t[1] + m_cameras[i].R[7]*t[2]);
			//m_cameras[i].t[2] = -(m_cameras[i].R[2]*t[0] + m_cameras[i].R[5]*t[1] + m_cameras[i].R[8]*t[2]);

			//std::cout<<i<<" "<<m_cameras[i].t[0]<<" "<<m_cameras[i].t[1]<<" "<<m_cameras[i].t[2]<<std::endl;
		}

		for (int i = 0 ; i < M; i++)
		{
			float pt[3];
			point_data[i].GetPoint(pt);

			m_point_data[i].m_pos[0] = pt[0];
			m_point_data[i].m_pos[1] = pt[1];
			m_point_data[i].m_pos[2] = pt[2];
		}

	}

	void CMultiCameraPnP::AdjustCurrentBundle(std::vector<ImageKeyVector> pt_views, bool _debug)
	{
		std::vector<CameraT> camera_data;
		std::vector<Point3D> point_data;
		std::vector<Point2D> measurements;
		std::vector<int> ptidx;
		std::vector<int> camidx;

		BundleAdjuster mba;

		SetBundleAdjustData(camera_data,point_data,measurements,ptidx,camidx,pt_views);

		if(!_debug)
		{
			mba.RunBundleAdjustment(camera_data,point_data,measurements,ptidx,camidx);

			GetBundleAdjustData(camera_data,point_data,measurements,ptidx,camidx);

			return;
		}
			
		//////////////////////////////////////////////////////////////////////////
		// compute projection error
		std::vector<float> reproErrs(curr_num_cameras,0);
		std::vector<int> counts(curr_num_cameras,0);

		for (int i =0; i < ptidx.size(); i++)
		{
			//std::cout<<i<<" "<<camidx[i]<<" "<<ptidx[i]<<std::endl;
			
			int idx = ptidx[i];
			Point3D point3d = point_data[idx];
			Point2D point2d = measurements[i];
			assert(camidx[i] >= 0 && camidx[i] < curr_num_cameras);

			CameraT cam = camera_data[camidx[i]];
			
			float focal_length = cam.GetFocalLength();
			double R[9] ;
			double t1[3],t2[3] ;
			double c[3];
			cam.GetMatrixRotation(R);
			cam.GetTranslation(t1);
			cam.GetCameraCenter(c);

			t2[0] = -(R[0]*c[0] + R[1]*c[1] + R[2]*c[2]);
			t2[1] = -(R[3]*c[0] + R[4]*c[1] + R[5]*c[2]);
			t2[2] = -(R[6]*c[0] + R[7]*c[1] + R[8]*c[2]);

			std::cout<<t2[0]<<" "<<t1[0]<<" "<<t2[1]<<" "<<t1[1]<<" "<<t2[2]<<" "<<t1[2]<<std::endl;

			cv::Mat_<double> K = (cv::Mat_<double>(3, 3)<<
								focal_length,	0.0,	0.0,
								0.0,	focal_length,	0.0,
								0.0,	0.0,	1.0);
		
			cv::Mat_<double> P = (cv::Mat_<double>(3,4) <<
									R[0], R[1], R[2], t2[0],
									R[3], R[4], R[5], t2[1],
									R[6], R[7], R[8], t2[2]);
			
			cv::Mat_<double> X = (cv::Mat_<double>(4,1)<< (point3d.xyz[0]),point3d.xyz[1],point3d.xyz[2],1.0);
			cv::Mat_<double> xt = -K*(P*X);

			float px =   xt(0)/xt(2);
			float py =	 xt(1)/xt(2);

			float dist = (px - point2d.x)*(px - point2d.x) + (py - point2d.y)*(py - point2d.y);

			reproErrs[camidx[i]] += sqrtf(dist);
			counts[camidx[i]] ++;

		}

		for (int i=0; i <curr_num_cameras; i++ )
		{
			cout<< counts[i] <<" "<<reproErrs[i]/counts[i]<<endl;
		}

	}

	// check the point_view is correct or not.
	int CMultiCameraPnP::CheckPointKeyConsistency(const std::vector<ImageKeyVector> pt_views,
		std::vector<int> added_order)
	{
		std::vector<ImageData> image_data = m_matchTracks.m_image_data;
		int num_points = pt_views.size();
		int errs = 0;
		for (int i = 0; i < num_points; i++ )
		{
			int num_views = pt_views[i].size();
			for (int j = 0; j < num_views; j++ )
			{
				int camId = pt_views[i][j].first;
				int im_key = pt_views[i][j].second;

				int im_idx = added_order[camId];

				int pt3d_idx = image_data[im_idx].m_keys[im_key].m_extra;

				if(pt3d_idx != i)
				{
					errs += 0;

				}
			}
		}

		printf("[CheckPointKeyConsistency] There were %d errors\n", errs);

		return errs;
	}

	void CMultiCameraPnP::SaveModelFile(const char* outpath,std::vector<ImageKeyVector> pt_views)
	{
		std::vector<CameraT> camera_data;
		std::vector<Point3D> point_data;
		std::vector<Point2D> measurements;
		std::vector<int> ptidx;
		std::vector<int> camidx;
		std::vector<int> keyidx;

		std::vector<string> names;
		std::vector<int> ptc;

		int N = curr_num_cameras;
		int M = m_point_data.size();

		int nproj = 0;
		camera_data.resize(N);
		names.resize(N);

		for (int i=0 ; i<N ; i++)
		{
			double q[9], t[3], d[2];

			double mf = m_cameras[i].f;
			camera_data[i].SetFocalLength(mf);

			camera_data[i].SetMatrixRotation(m_cameras[i].R);
			// -R(-R^T*t) = RR^T*t = t
			//t[0] = -(m_cameras[i].R[0]*m_cameras[i].t[0] + m_cameras[i].R[1]*m_cameras[i].t[1] + m_cameras[i].R[2]*m_cameras[i].t[2]);
			//t[1] = -(m_cameras[i].R[3]*m_cameras[i].t[0] + m_cameras[i].R[4]*m_cameras[i].t[1] + m_cameras[i].R[5]*m_cameras[i].t[2]);
			//t[2] = -(m_cameras[i].R[6]*m_cameras[i].t[0] + m_cameras[i].R[7]*m_cameras[i].t[1] + m_cameras[i].R[8]*m_cameras[i].t[2]);

			//设置参数就是相机的空间坐标
			camera_data[i].SetTranslation(m_cameras[i].t);

			d[0] = d[1] = 0;
			camera_data[i].SetNormalizedMeasurementDistortion(d[0]);

			int camera_order = good_views[i];
			names[i] = m_feature_list[camera_order];
		}

		int npoint = M;
		point_data.resize(npoint);

		for (int i = 0 ; i < npoint; i++)
		{
			float pt[3];

			pt[0] = m_point_data[i].m_pos[0];
			pt[1] = m_point_data[i].m_pos[1];
			pt[2] = m_point_data[i].m_pos[2];

			point_data[i].SetPoint(pt); 

			//color 
			int* color = m_point_data[i].m_color;
			ptc.push_back(color[0]);
			ptc.push_back(color[1]);
			ptc.push_back(color[2]);

			//2d projection
			for (unsigned int k =0; k < pt_views[i].size(); k++) 
			{
				ImageKey ikey = pt_views[i][k];
				int cam_id = ikey.first;
				int key_idx = ikey.second;

				int im_idx = good_views[cam_id];
				float x = (GetImageKey(im_idx,key_idx).m_x - m_cx );
				float y = (GetImageKey(im_idx,key_idx).m_y - m_cy );

// 				cv::Mat_<double> matP = (cv::Mat_<double>(3,1)<< -x,-y,1.0);
// 				double K[9];
// 				GetIntrinsics(m_cameras[cam_id],K);
// 				cv::Mat mK = cv::Mat(3,3,CV_64F,K);
// 				cv::Mat_<double> mp = mK.inv()*matP;
// 
// 				cv::Point3d point_norm = mp.at<cv::Point3d>(0);
				//add a measurment to the vector
				measurements.push_back(Point2D(-x,-y));

				camidx.push_back(cam_id);    //camera index
				ptidx.push_back(i);          //point index
				keyidx.push_back(key_idx);   // keypoint idx
				nproj ++;
			}		
		}

		///////////////////////////////////////////////////////////////////////////////
		std::cout << N << " cameras; " << npoint << " 3D points; " << nproj << " projections\n";

		if(outpath == NULL) return;
		if(strstr(outpath, ".nvm"))
			SaveNVM(outpath, camera_data, point_data, measurements, ptidx, camidx,keyidx, names, ptc); 

	}

#define Vx(v) (v)[0]
#define Vy(v) (v)[1]
#define Vz(v) (v)[2]

	void CMultiCameraPnP::DumpOutputFile(const char *output_dir, const char *filename, 
		int num_points,	std::vector<int> order, 
		camera_params_t *cameras, 
		std::vector<PointData>& points,
		std::vector<ImageKeyVector> &pt_views)
	{
		int num_visible_points = 0;
		for (int i = 0; i < num_points; i++) {
			if (pt_views[i].size() > 0)
				num_visible_points++;
		}

		char buf[256];
		sprintf(buf, "%s/%s", output_dir, filename);

		FILE *f = fopen(buf, "w");
		if (f == NULL) {
			printf("Error opening file %s for writing\n", buf);
			return;
		}

		/* Print version number */
		fprintf(f, "# Bundle file v0.3\n");
		/* Print number of cameras and points */
		fprintf(f, "%d %d\n", n_images, num_visible_points);

		/* Dump cameras */
		for (int i = 0; i < n_images; i++) {

			int idx = -1;
			for (int j = 0; j < curr_num_cameras; j++) {
				if (order[j] == i) {
					idx = j;
					break;
				}
			}

			if (idx == -1) {
				fprintf(f, "0 0 0\n");
				fprintf(f, "0 0 0\n0 0 0\n0 0 0\n0 0 0\n");
			} else {
				fprintf(f, "%0.10e %0.10e %0.10e\n", 
					cameras[idx].f, cameras[idx].k[0], cameras[idx].k[1]);

				fprintf(f, "%0.10e %0.10e %0.10e\n", 
					cameras[idx].R[0],cameras[idx].R[1], cameras[idx].R[2]);
				fprintf(f, "%0.10e %0.10e %0.10e\n", 
					cameras[idx].R[3],cameras[idx].R[4], cameras[idx].R[5]);
				fprintf(f, "%0.10e %0.10e %0.10e\n", 
					cameras[idx].R[6],cameras[idx].R[7], cameras[idx].R[8]);

				double t[3];
				//matrix_product(3, 3, 3, 1, cameras[idx].R, cameras[idx].t, t);
				//matrix_scale(3, 1, t, -1.0, t);
				t[0] = -(cameras[idx].R[0]*cameras[idx].t[0] + cameras[idx].R[1]*cameras[idx].t[1] + cameras[idx].R[2]*cameras[idx].t[2]);
				t[1] = -(cameras[idx].R[3]*cameras[idx].t[0] + cameras[idx].R[4]*cameras[idx].t[1] + cameras[idx].R[5]*cameras[idx].t[2]);
				t[2] = -(cameras[idx].R[6]*cameras[idx].t[0] + cameras[idx].R[7]*cameras[idx].t[1] + cameras[idx].R[8]*cameras[idx].t[2]);

				fprintf(f, "%0.10e %0.10e %0.10e\n", t[0], t[1], t[2]);
			}
		}

		/* Dump points */
		for (int i = 0; i < num_points; i++) {
			int num_visible = (int) pt_views[i].size();

			if (num_visible > 0) {

				/* Position */
				fprintf(f, "%0.10e %0.10e %0.10e\n", 
					Vx(points[i].m_pos), Vy(points[i].m_pos), Vz(points[i].m_pos));

				/* Color */
				fprintf(f, "%d %d %d\n", 
					Vx(points[i].m_color), 
					Vy(points[i].m_color), 
					Vz(points[i].m_color));

				int num_visible = (int) pt_views[i].size();
				fprintf(f, "%d", num_visible);
				for (int j = 0; j < num_visible; j++) {
					int img = order[pt_views[i][j].first];
					int key = pt_views[i][j].second;

					double x = - (GetImageKey(img,key).m_x - m_cx);
					double y = - (GetImageKey(img,key).m_y - m_cy);

					fprintf(f, " %d %d %0.4f %0.4f", img, key, x, y);
				}

				fprintf(f, "\n");
			}
		}

		fclose(f);
	}

	void CMultiCameraPnP::Bundler2PMVS(std::vector<ImageKeyVector> pt_view)
	{
		char filename[128] = "bundle.out";
		DumpOutputFile(m_path,filename,m_point_data_index,good_views,m_cameras,m_point_data,pt_view);

		sprintf(filename, "%s/list.txt", m_path);
		FILE *f = fopen(filename, "w");
		for (int i =0; i < n_images; i++)
		{
			fprintf(f,"%s\n",m_feature_list[i].c_str());
		}

		fclose(f);

		// cameras 总数与图片相等，利用good_views的编号初始化对应相机内参
		std::vector<camera_params_t> cameras;
		cameras.resize(n_images);
		std::vector<int> added_order;
		std::vector<int> map(n_images);
		int k = 0;
		for (int i = 0; i < n_images; i++)
		{
			std::vector<int>::iterator it = find(good_views.begin(),good_views.end(),i);
			if(it == good_views.end())
			{
				InitializeCameraParams(cameras[i]);
				map[i] = -1;
// 				cameras[i].R[0] = 1.0;  cameras[i].R[1] = 0.0;  cameras[i].R[2] = 0.0;
// 				cameras[i].R[3] = 0.0;  cameras[i].R[4] = 1.0;  cameras[i].R[5] = 0.0;
// 				cameras[i].R[6] = 0.0;  cameras[i].R[7] = 0.0;  cameras[i].R[8] = 1.0;
// 				cameras[i].t[0] = 0.0;
// 				cameras[i].t[1] = 0.0;
// 				cameras[i].t[2] = 0.0;
			}
			else
			{
				int idx = (it - good_views.begin());
				cameras[i] = m_cameras[idx];
				added_order.push_back(idx);
				map[i] = k;
				k++;
			}
		}
		// 每张图片对应的添加顺序【added_order[0] 表示第一张图片在good_view中的位置 】
		for (int i = 0; i < added_order.size(); i++)
			std::cout<<added_order[i]<<" ";
		std::cout<<std::endl;
		// good_view 添加的顺序 【good_views[0] 表示第一次添加的是哪一幅图片 】
		for (int i = 0; i < good_views.size(); i++)
			std::cout<<i<<" "<<good_views[i]<<", ";
		std::cout<<std::endl;

		// 
		std::vector<float> reproErrs(curr_num_cameras,0);
		std::vector<int> counts(curr_num_cameras,0);
		for(int i = 0; i < m_point_data_index ; i++)
		{
			
			PointData point = m_point_data[i];
			cv::Mat_<double> X = (cv::Mat_<double>(4,1)<<Vx(point.m_pos),Vy(point.m_pos),Vz(point.m_pos),1.0);

			std::vector<view_t> m_views = point.m_views;
			for(int j = 0 ; j < m_views.size(); j++)
			{
				int image = map[m_views[j].image] ;// 按原始照片列表顺序		
				
				if(image < 0 || image >= curr_num_cameras)
				{
					std::cout<<"======================error1=============="<<std::endl;
					break;
				}
				camera_params_t cam = cameras[m_views[j].image];
				
				cv::Mat_<double> K = (cv::Mat_<double>(3, 3)<<
									cam.f,	0.0,	0.0,
									0.0,	cam.f,	0.0,
									0.0,	0.0,	1.0);
				double *R = cam.R;
				double *c = cam.t;  //camera center
				double t[3];
				t[0] = -(R[0]*c[0] + R[1]*c[1] + R[2]*c[2]);
				t[1] = -(R[3]*c[0] + R[4]*c[1] + R[5]*c[2]);
				t[2] = -(R[6]*c[0] + R[7]*c[1] + R[8]*c[2]);

				cv::Mat_<double> P = (cv::Mat_<double>(3,4) <<
									  R[0], R[1], R[2], t[0],
									  R[3], R[4], R[5], t[1],
									  R[6], R[7], R[8], t[2]);
				
				cv::Mat_<double> xt = K*(P*X);

				float px =   xt(0)/xt(2);
				float py =	 xt(1)/xt(2);

				int im_idx = m_views[j].image;
				int key_idx = m_views[j].key;

				int x = GetImageKey(im_idx,key_idx).m_x - m_cx;
				int y = GetImageKey(im_idx,key_idx).m_y - m_cy;

				float dist = (-x - px)*(-x - px) + (-y - py)*(-y - py);
				//cout<<x<<" "<<y<<" "<<px<<" "<<py<<endl;

				reproErrs[image] += sqrtf(dist);
				counts[image]++;
			}
		}

		for (int i=0; i <curr_num_cameras; i++ )
		{
			cout<< counts[i] <<" "<<reproErrs[i]/counts[i]<<endl;
		}

		Utils utl;
		utl.WritePMVS(m_path,m_feature_list,cameras,added_order,m_point_data);
	}

	CMultiCameraPnP::~CMultiCameraPnP()
	{
		delete [] m_cameras;
	}
}
