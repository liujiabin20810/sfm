
#include "MultiCameraPnP.h"
#include "BundleAdjuster.h"

#include <windows.h>
#include <direct.h>
#include <io.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <fstream>

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

		std::cout << "Path : " << dir_name_ << std::endl;
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
		if (fs.open(std::string(strPath) + "\\out_camera_data.yml", cv::FileStorage::READ)) {
			fs["camera_matrix"] >> cam_matrix;
			fs["distortion_coefficients"] >> m_distortion_coeff;
		}
		else {
			//no calibration matrix file - mockup calibration
			int image_width= m_images[0].cols;
			int image_hight = m_images[0].rows;
			double max_w_h = MAX(image_hight, image_width);
			cam_matrix = (cv::Mat_<double>(3, 3) << max_w_h, 0, image_width / 2.0,
				0, max_w_h, image_hight / 2.0,
				0, 0, 1);
			m_distortion_coeff = cv::Mat_<double>::zeros(1, 4);
		}

		m_K = cam_matrix;
		cv::invert(m_K, m_Kinv); //get inverse of camera matrix

		m_distortion_coeff.convertTo(m_distcoeff_32f, CV_32FC1);
		m_K.convertTo(m_K_32f, CV_32FC1);

		return 0;
	}

	void CMultiCameraPnP::initCalibMatrix(cv::Mat _K, cv::Mat _distortion_coeff)
	{
		K = _K.clone();
		distortion_coeff = _distortion_coeff.clone();

		cv::invert(K, Kinv); //get inverse of camera matrix

		_distortion_coeff.convertTo(distcoeff_32f, CV_32FC1);

		K.convertTo(K_32f, CV_32FC1);
	
		m_point_data_index = 0;
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
		for (int i = 0; i<10; i++) { //Hartley suggests 10 iterations at most
			cv::Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
			X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

			//recalculate weights
			double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
			double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

			//breaking point
			if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

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
		cv::Matx34d P = Pmats[ith_camera];
		cv::Matx34d P1 = Pmats[jth_camera];

		std::vector<cv::KeyPoint> pt_set1 = images_points[ith_camera];
		std::vector<cv::KeyPoint> pt_set2 = images_points[jth_camera];

		MatchIndex idx = GetMatchIndex(ith_camera,jth_camera);
		std::vector<KeypointMatch>& matchList = m_matchTracks.m_matches_table.GetMatchList(idx);
		
		unsigned int match_size = matchList.size();
		std::cout<<"[TriangulatePoints] matchSize= "<< match_size <<std::endl;

		cv::Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
			P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
			P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
			0, 0, 0, 1);

		cv::Matx44d P1inv(P1_.inv());
		cv::vector<double> reproj_error;
		cv::Mat_<double> KP1 = K * cv::Mat(P1);

		//#pragma omp parallel for num_threads(4)
		for (int i = 0; i < match_size; i++) {
			
			int queryIdx = matchList[i].m_idx1;

			cv::Point2f kp = pt_set1[queryIdx].pt;
			cv::Point3d u(kp.x, kp.y, 1.0);
			cv::Mat_<double> um = Kinv * cv::Mat_<double>(u);
			u.x = um(0); u.y = um(1); u.z = um(2);
			
			int trainIdx = matchList[i].m_idx2;
			cv::Point2f kp1 = pt_set2[trainIdx].pt;
			cv::Point3d u1(kp1.x, kp1.y, 1.0);
			cv::Mat_<double> um1 = Kinv * cv::Mat_<double>(u1);
			u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

			cv::Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);

			cv::Mat_<double> xPt_img = KP1 * X;

			cv::Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

			//#pragma omp critical
			{
				double reprj_err = cv::norm(xPt_img_ - kp1);
				reproj_error.push_back(reprj_err);
				
				PointData cp;
				cp.m_pos[0] = X(0);
				cp.m_pos[1] = X(1);
				cp.m_pos[2] = X(2);
				
				cp.m_views.push_back(MatchIndex(ith_camera,queryIdx));
				cp.m_views.push_back(MatchIndex(jth_camera,trainIdx));
				cp.m_ref_image = ith_camera;

				pointcloud.push_back(cp);
			}
		}

		cv::Scalar me = cv::mean(reproj_error);

		return me[0];
	}

	bool CMultiCameraPnP::TriangulatePointsBetweenViews(int working_view,int older_view)
	{

		std::vector<PointData> new_triangulated;

		//adding more triangulated points to general cloud
		double reproj_error = TriangulatePoints(older_view,working_view,new_triangulated);
	
//#ifdef _DEBUG
		std::cout << "[TriangulatePointsBetweenViews] triangulation reproj error " << reproj_error<<" /"<<new_triangulated.size()<< std::endl;
//#endif //_DEBUG

		if(reproj_error > 20.0) 
		{
			// somethign went awry, delete those triangulated points
			std::cerr << "[TriangulatePointsBetweenViews] reprojection error."<<std::endl;
			return false;
		}
		
		std::vector<PointData>::iterator p3d_iter =  new_triangulated.begin();

		//for(int i=0; i < new_triangulated.size(); i++)
		int pt3d_num = 0;
		for(;p3d_iter != new_triangulated.end(); ++p3d_iter )
		{
			
			int im_index = p3d_iter->m_views[0].first;
			int im_key  = p3d_iter->m_views[0].second;

			//std::cout<<"[image info ]"<<im_index<<" "<< im_key <<std::endl;
			ImageData image_data = m_matchTracks.m_image_data[im_index];
			/*std::pair<std::vector<int>::const_iterator,std::vector<int>::const_iterator> p;
			const std::vector<int> pt1 = image_data.m_visible_keys;
			// equal_range 不能用于非顺序数组
			p = equal_range(pt1.begin(), pt1.end(), im_key);
			assert( p.first != p.second);*/
			std::vector<int> pt1 = image_data.m_visible_keys;

			std::vector<int>::iterator result = std::find( pt1.begin( ), pt1.end( ), im_key ); 

			int offset = result - pt1.begin( );
			assert(offset != pt1.size());
			// 当前点对应的 track_data
			
			int k = image_data.m_visible_points[offset];
			//std::cout<<"[track index] "<<offset<<" "<<k<<std::endl;
			if( m_matchTracks.m_track_data[k].m_extra == -1)
			{
				m_matchTracks.m_track_data[k].m_extra = m_point_data_index;
				p3d_iter->m_num_vis = m_matchTracks.m_track_data[k].m_views.size();

				//add new cloud points
				m_point_data.push_back(*p3d_iter);
				m_point_data_index++;

				pt3d_num++;
			}
		}

		std::cout<<"[TriangulatePointsBetweenViews] new pt3d num: "<<pt3d_num <<" totle pt3d num: "<<m_point_data_index<<std::endl;

		return true;
	}

	bool CMultiCameraPnP::match()
	{
		// calculate siftgpu feature and matches
		if(!calc_sift_feature_and_match())
		{
			std::cerr<<"calc sift feature and match err."<<std::endl;
			return false;
		}

		// compute tracks
		m_matchTracks.InitMatchTable(matches_matrix,images_points,F_matrix,m_images,n_images);

		m_matchTracks.ComputeGeometricConstraints();

		/* Set track pointers to -1 */
		for (int i = 0; i < (int) m_matchTracks.m_track_data.size(); i++) {
			m_matchTracks.m_track_data[i].m_extra = -1;
		}

		return true;
	}

	bool sort_by_first(std::pair<int, std::pair<int, int> > a, std::pair<int, std::pair<int, int> > b) { return a.first > b.first; }
	bool sort_by_second(std::pair<MatchIndex,int> a,std::pair<MatchIndex,int> b) { return a.second > b.second ;}
		
	bool CMultiCameraPnP::GetBaseLineTriangulation()
	{
#ifdef _DEBUG
		std::cout << "=========================== Baseline triangulation ===========================\n";
#endif // _DEBUG

		cv::Matx34d P(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0),
			P1(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);

		std::vector<CloudPoint> tmp_pcloud;
		std::list<std::pair<MatchIndex,int  > > matches_sizes;
		MatchTable& m_matches = m_matchTracks.m_matches_table;

		for(int i=0; i < n_images-1;i++)
			for(int j= i+1; j <n_images; j++ )
			{
				MatchIndex idx = GetMatchIndex(i,j);

				int _sz = m_matches.GetMatchList(idx).size();
				//printf("[%d,%d]:%d\n",i,j,_sz);
				matches_sizes.push_back(std::make_pair(idx,_sz));
			}

		matches_sizes.sort(sort_by_second);

		m_first_view = m_second_view = 0;

		std::list<std::pair<MatchIndex,int > >::iterator highest_pair = matches_sizes.begin();
		
		bool goodF = false;
		for (; highest_pair != matches_sizes.end() && !goodF; ++highest_pair)
		{
			std::cout << "[" << highest_pair->first.first << "," << highest_pair->first.second << "]: " << highest_pair->second << std::endl;
		
			m_second_view = (*highest_pair).first.second;
			m_first_view = (*highest_pair).first.first;

			goodF = FindCameraMatrices(K, Kinv, distortion_coeff,
				images_points[m_first_view],
				images_points[m_second_view],
				m_first_view,
				m_second_view,
				F_matrix[std::make_pair(m_first_view, m_second_view)],
				P,
				P1);

			if (goodF)
			{
				Pmats[m_first_view] = P;
				Pmats[m_second_view] = P1;	
				bool good_triangulation = TriangulatePointsBetweenViews(m_second_view, m_first_view);
			
				if (!good_triangulation)
				{
					goodF = false;
					Pmats[m_first_view] = 0;
					Pmats[m_second_view] = 0;
				}
			}

		}


		if (!goodF) {
			std::cerr << "Cannot find a good pair of images to obtain a baseline triangulation" << std::endl;
			return false;
		}

		return true;
	}

	// Find P matrix
	bool CMultiCameraPnP::FindCameraMatrices(const cv::Mat& K,
		const cv::Mat& Kinv,
		const cv::Mat& distcoeff,
		const cv::vector<cv::KeyPoint>& imgpts1,
		const cv::vector<cv::KeyPoint>& imgpts2,
		int ith_camera,
		int jth_camera,
		cv::Mat F,
		cv::Matx34d& P,
		cv::Matx34d& P1)
	{
		std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(ith_camera,jth_camera)];

		if (matches.size() < 100)
		{
			std::cerr << "not enough inliers after F matrix" << std::endl;
			return false;
		}

		//Essential matrix: compute then extract cameras [R|t]
		cv::Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

		if (fabs(determinant(E)) > 1e-07) {
			std::cout << "det(E) != 0 : " << determinant(E) << std::endl;
			P1 = 0;
			return false;
		}

		//decompose E to P' , HZ (9.19)
		cv::SVD svd(E, cv::SVD::MODIFY_A);
		cv::Mat svd_u = svd.u;
		cv::Mat svd_vt = svd.vt;
		cv::Mat svd_w = svd.w;
		cv::Matx33d W(0, -1, 0,//HZ 9.13
			1, 0, 0,
			0, 0, 1);

		cv::Mat_<double> R = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
		cv::Mat_<double> t = svd_u.col(2); //u3

		if (fabsf(determinant(R)) - 1.0 > 1e-07) {
			std::cerr << "det(R) != +-1.0, this is not a rotation matrix" << std::endl;
			P1 = 0;
			return false;
		}


		P1 = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
			R(1, 0), R(1, 1), R(1, 2), t(1),
			R(2, 0), R(2, 1), R(2, 2), t(2));

		return true;
	}

	void CMultiCameraPnP::RecoverDepthFromImages()
	{
		std::cout << "======================================================================\n";
		std::cout << "======================== Depth Recovery Start ========================\n";
		std::cout << "======================================================================\n";

		GetBaseLineTriangulation();
		WriteCloudPoint();
		AdjustCurrentBundle();

		cv::Matx34d P1 = Pmats[m_second_view];
		cv::Mat_<double> t = (cv::Mat_<double>(1,3) << P1(0,3), P1(1,3), P1(2,3));
		cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0), P1(0,1), P1(0,2), 
														P1(1,0), P1(1,1), P1(1,2), 
														P1(2,0), P1(2,1), P1(2,2));
		cv::Mat_<double> rvec(1,3); Rodrigues(R, rvec);
	
		done_views.insert(m_first_view);
		done_views.insert(m_second_view);
		good_views.insert(m_first_view);
		good_views.insert(m_second_view);

		cv::namedWindow("image_match",1);
		while (done_views.size() != n_images)
		{
			std::cout<<"[LOOP ]cloud size: "<<m_point_data_index<<std::endl;
			//find image with highest 2d-3d correspondance [Snavely07 4.2]
			unsigned int max_2d3d_view = -1, max_2d3d_count = 0;
			for (int _i=0; _i < n_images; _i++) 
			{				
				if(done_views.find(_i) != done_views.end()) 
					continue; //already done with this view				
				
				std::vector<int> & tracks = m_matchTracks.m_image_data[_i].m_visible_points;

				int _sz = tracks.size();
				int used_track_num = 0;
				for(int j=0; j<_sz; j++)
				{
					int k = tracks[j];
					if(m_matchTracks.m_track_data[k].m_extra >= 0 )
						used_track_num++;
				}

				if(max_2d3d_count < used_track_num)
				{
					max_2d3d_count = used_track_num;
					max_2d3d_view = _i;
				}
			}

			std::cout<<"[RecoverDepthFromImages] Matched View: "<<max_2d3d_view<<" Matched Count: "<<max_2d3d_count<<std::endl;
			int i = max_2d3d_view; //highest 2d3d matching view
			done_views.insert(i);

			bool pose_estimated = FindPoseEstimation(i,rvec,t,R);
			if(!pose_estimated)
				continue;
			
			//store estimated pose	
			Pmats[i] = cv::Matx34d	(R(0,0),R(0,1),R(0,2),t(0),
								 R(1,0),R(1,1),R(1,2),t(1),
								 R(2,0),R(2,1),R(2,2),t(2));


			// start triangulating with previous GOOD views
			std::set<int>::iterator done_view = good_views.begin();
			for (; done_view != good_views.end(); ++done_view) 
			{
				int view = *done_view;
				if( view == i )
					continue; //skip current...

				//show image match

				cv::Mat drawImage = drawImageMatches(i,view);
				cv::imshow("image_match",drawImage);
				cv::waitKey();
				//
				int cloud_size_befor_triangulation = m_point_data_index;

				bool good_triangulation = TriangulatePointsBetweenViews(i,view);
				
				if(!good_triangulation) 
					continue;

				int cloud_size_after_triangulation = m_point_data_index;

				std::cout<<"[RecoverDepthFromImages] Added cloud points: " \
					<<(cloud_size_after_triangulation - cloud_size_befor_triangulation)<<std::endl;

				AdjustCurrentBundle();
			}

			good_views.insert(i);

		}

		AdjustCurrentBundle();

		std::cout << "======================================================================\n";
		std::cout << "========================= Depth Recovery DONE ========================\n";
		std::cout << "======================================================================\n";
	}

	bool CMultiCameraPnP::FindPoseEstimation(int working_view,cv::Mat_<double>& rvec,
				cv::Mat_<double>& t,cv::Mat_<double>& R)
	{
		std::vector<cv::Point3f> ppcloud;
		std::vector<cv::Point2f> imgpoints;

		Find2D3DCorrespondences(working_view,ppcloud,imgpoints);

		if(ppcloud.size() <= 7 || imgpoints.size() <= 7 || ppcloud.size() != imgpoints.size())
		{ 
			//something went wrong aligning 3D to 2D points..
			std::cerr << "[FindPoseEstimation] Couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<std::endl;
			return false;
		}
		
		std::vector<int> inliers;

		double minVal,maxVal;
		cv::minMaxIdx(imgpoints,&minVal,&maxVal);
	
		cv::solvePnPRansac(ppcloud, imgpoints, K, distortion_coeff, rvec, t, true, 1000, 10.0, 0.25 * (double)(imgpoints.size()), inliers, CV_ITERATIVE);
	
		std::vector<cv::Point2f> projected3D;
		cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);
		
		std::cout<<"[FindPoseEstimation] ("<<inliers.size()<<"/"<<imgpoints.size()<<")"<<std::endl;

		//get inliers
		if(inliers.size()==0)
		{
			for(int i=0;i<projected3D.size();i++) {
				if(cv::norm(projected3D[i]-imgpoints[i]) < 10.0)
					inliers.push_back(i);
			}
		 }

		if(inliers.size() < (imgpoints.size())/5.0)
		{
			std::cerr << "[FindPoseEstimation] Not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgpoints.size()<<")"<< std::endl;
			return false;
		}

		if(cv::norm(t) > 200.0) 
		{
		// this is bad...
			std::cerr << "[FindPoseEstimation] Estimated camera movement is too big, skip this camera\r\n";
			return false;
		}

		cv::Rodrigues(rvec, R);
		if(fabsf(determinant(R))-1.0 > 1e-07)
		{
			std::cerr << "[FindPoseEstimation] Rotation is incoherent. we should try a different base view..." << std::endl;
			return false;
		}

		return true;
	}

	void CMultiCameraPnP::Find2D3DCorrespondences(int working_view, 
	std::vector<cv::Point3f>& ppcloud, 
	std::vector<cv::Point2f>& imgPoints)
	{
		// 需要通过已知的2d->3d 的关系以及 2d->2d 的关系，
		// 获得新的 2d->3d 的关系
		ppcloud.clear();
		imgPoints.clear();

		//获取在当前帧中，点云匹配的二维点
		ImageData working_imagedata = m_matchTracks.m_image_data[working_view];
		std::vector<TrackData> track_data =  m_matchTracks.m_track_data;

		std::vector<cv::KeyPoint> keypoints = m_matchTracks.m_images_points[working_view];

		int track_num = working_imagedata.m_visible_points.size();
		for(int i =0; i< track_num; i++)
		{
			int k = working_imagedata.m_visible_points[i];

			if(track_data[k].m_extra >= 0)
			{
				cv::Point3f pt3d;
				pt3d.x = m_point_data[track_data[k].m_extra].m_pos[0];
				pt3d.y = m_point_data[track_data[k].m_extra].m_pos[1];
				pt3d.z = m_point_data[track_data[k].m_extra].m_pos[2];

				ppcloud.push_back(pt3d);

				int pt2d_index = working_imagedata.m_visible_keys[i];

				cv::Point2f pt2d ;
				pt2d.x = keypoints[pt2d_index].pt.x;
				pt2d.y = keypoints[pt2d_index].pt.y;
					
				imgPoints.push_back(pt2d);
			}
		}
	}

	void CMultiCameraPnP::WriteCloudPoint()
	{
		std::ofstream fp("cloud.txt");

		if(fp.is_open())
		{
			for(int i=0; i < m_point_data_index; i++)
			{
				double *point = m_point_data[i].m_pos;
				fp<< point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;
			}
		}
		fp.close();

	}

	void CMultiCameraPnP::AdjustCurrentBundle()
	{
		BundleAdjuster BA;
		BA.adjustBundle(m_point_data,K,images_points,Pmats);
	}

	CMultiCameraPnP::~CMultiCameraPnP()
	{

	}
}
