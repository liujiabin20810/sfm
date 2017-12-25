
#include "FeatureMatcher.h"
#include "SIFTGPU.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
namespace bundler
{

	// init image masks
	void CPruneFeature::init_image_mask(std::vector<cv::Mat> image_masks)
	{
		int mask_size = image_masks.size();

		assert(mask_size == n_images);

		for(int i = 0; i < mask_size; i++)
		{
			cv::Mat im = image_masks[i];
			assert(!im.empty());
			cv::Mat gray;
			cv::cvtColor(im,gray,CV_BGR2GRAY);

			cv::Mat mask = gray > 5;

			cv::imshow("mask",mask);
			cv::waitKey();
			m_image_masks.push_back(mask);
		}
	}
	//////////////////////////////////////////////////////////////////////////
	CFeatureMatcher::CFeatureMatcher()
	{
		m_sift = new SIFT(4096);

		char * v[] = { "-fo", "-1", "-v", "0" };//
		m_sift->ParseParam(4, v);

		for (size_t i = 0; i < vec_descriptors.size(); i++)
			vec_descriptors[i].clear();
		vec_descriptors.clear();

		for (size_t i = 0; i < images_points.size(); i++)
			images_points[i].clear();
		images_points.clear();

		m_init_images_list = false;
	}

	CFeatureMatcher::CFeatureMatcher(std::vector<std::string> images_list,std::vector<cv::Mat> images)
	{
		m_sift = new SIFT(4096);

		char * v[] = { "-fo", "-1", "-v", "0" };//
		m_sift->ParseParam(4, v);

		assert(images_list.size() > 0);

		m_sift->setImageList(images_list);

		n_images = images_list.size();
		for(int i=0; i < n_images; i++)
			m_images.push_back(images[i]);

		m_init_images_list = true;

		for (size_t i = 0; i < vec_descriptors.size(); i++)
			vec_descriptors[i].clear();
		vec_descriptors.clear();

		for (size_t i = 0; i < images_points.size(); i++)
			images_points[i].clear();
		images_points.clear();

	}

	void CFeatureMatcher::init_images_list(std::vector<std::string> images_list)
	{

		m_sift = new SIFT(4096);

		char * v[] = { "-fo", "-1", "-v", "0" };//
		m_sift->ParseParam(4, v);

		assert(images_list.size() > 0);

		m_sift->setImageList(images_list);

		n_images = images_list.size();
		m_init_images_list = true;

		for (size_t i = 0; i < vec_descriptors.size(); i++)
			vec_descriptors[i].clear();
		vec_descriptors.clear();

		for (size_t i = 0; i < images_points.size(); i++)
		{
			images_points[i].clear();
			images_points_colors[i].clear();
		}

		images_points.clear();
		images_points_colors.clear();
	}
	
	// init with images
	bool CFeatureMatcher::calc_sift_feature(std::vector<cv::Mat> images)
	{
		assert(images.size() > 0);

		n_images = images.size();

		for (int i = 0; i < n_images; i++)
		{
			std::vector<SiftKeypoint> keypoints;
			std::vector<float> descriptors;

			//SIFTGPU method
			(*m_sift)(images[i],keypoints, descriptors);

			std::vector<cv::Point2f> points;
			m_sift->convertSiftKeypoint(keypoints, points);

			std::vector<cv::KeyPoint> cvkeypoints;
			cv::KeyPoint::convert(points, cvkeypoints);

			images_points.push_back(cvkeypoints);

			vec_descriptors.push_back(descriptors);
			
		}

		return n_images > 0;
	}
	// init with images_list
	bool CFeatureMatcher::calc_sift_feature_and_match()
	{
		if (!m_init_images_list)
		{
			std::cerr<<"[calc_sift_feature_and_match]have no images to calc sift feature. "<<std::endl;
			return false;
		}

		std::vector<std::vector<SiftKeypoint>> keypoints;
		std::vector<std::vector<float> > descriptors;
		//SIFTGPU method
		// 1. get all images features
		// 2. get each good match on F matrix
		// 3. Save matches and F matrix
		(*m_sift)(keypoints, descriptors);

		m_sift->writeSiftFeature(keypoints,descriptors);
		//m_sift->writeLoweSiftFeature(keypoints,descriptors);

		std::vector<SiftKeypoint> queryKpts, trainedKpts;
		std::vector<float> queryDes, trainedDes;

		for (size_t i = 0; i < n_images - 1 ; i++)
		{
			queryKpts = keypoints[i];
			queryDes = descriptors[i];

			int n = queryKpts.size();
			int(*_buf)[2] = new int[n][2];

			for (size_t j = i + 1; j < n_images; j++)
			{
				trainedKpts = keypoints[j];
				trainedDes = descriptors[j];
				cv::Mat F;
				//std::cout<<"[MatchIndex] :"<<i<<","<<j<<std::endl;
				
				int  good_match_num = m_sift->match(queryKpts, trainedKpts, queryDes, trainedDes, _buf ,F);

				if (good_match_num == 0)
					continue;

				std::vector<cv::DMatch> _match;

				for (int index = 0; index < good_match_num; ++index)
				{
					int queryIndex = _buf[index][0];
					int trainedIndex = _buf[index][1];

					cv::DMatch _dm(queryIndex, trainedIndex, 0.0);
					_match.push_back(_dm);
				}

				std::pair<int, int> _index_pair = std::make_pair(i, j);
				matches_matrix[_index_pair] = _match;
				F_matrix[_index_pair] = F;
			} // j
		} // i
		
		//m_sift->writeSiftMatch(matches_matrix);

		for (size_t i = 0; i < n_images; i++)
		{
			std::vector<cv::Point2f> points;
			m_sift->convertSiftKeypoint(keypoints[i], points);

			std::vector<cv::KeyPoint> cvkeypoints;
			cv::KeyPoint::convert(points, cvkeypoints);

			images_points.push_back(cvkeypoints);
			vec_descriptors.push_back(descriptors[i]);
		}

		// feature match is good 
// 		for(int i = 0; i < n_images-1; i++)
// 			for(int j= i+1; j < n_images;j++){
// 			cv::Mat drawImage = drawImageMatches(i,j);
// 			cv::imshow("image_match",drawImage);
// 			cv::waitKey();
// 		}
		std::cout<<"[calc_sift_feature_and_match] Complete Matching."<<std::endl;
		return n_images > 0 ;
	}
	//
	bool CFeatureMatcher::calc_sift_feature()
	{
		if (!m_init_images_list)
		{
			std::cerr<<"[calc_sift_feature_and_match]have no images to calc sift feature. "<<std::endl;
			return false;
		}

		std::vector<std::vector<SiftKeypoint>> keypoints;
		std::vector<std::vector<float> > descriptors;
		//SIFTGPU method
		//get all images features
		(*m_sift)(keypoints, descriptors);

		return (m_sift->writeSiftFeature(keypoints,descriptors));
	}

	bool CFeatureMatcher::read_sift_feature()
	{
		char* filename = "crazyhorse/P1000966.sift";

		std::ifstream fin(filename,std::ios_base::binary); 
		if(!fin.is_open())
			return false;
 
		fin.seekg(0,  std::ios::end);   
		std::cout   <<   "file   length   = "   <<   fin.tellg()   <<   std::endl;   
		fin.seekg(std::ios::beg);     //   restore   saved   position

		std::vector<SiftKeypoint> keypoints;
		std::vector<float> descs;
		m_sift->readSiftFeature(fin,keypoints,descs);

		fin.close();

		char * outfile = "crazyhorse/P1000966.key";
		std::ofstream fout(outfile);

		if(!fout.is_open())
			return false;

		m_sift->writeLoweSiftFeature(fout,keypoints,descs);

		fout.close();

		return true;
	}

	cv::Mat CFeatureMatcher::drawImageMatches(int _index_i, int _index_j)
	{
		cv::Mat image_l = m_images[_index_i];
		cv::Mat image_r = m_images[_index_j];

		assert(image_l.type() == image_r.type());
		int totalCols = image_l.cols + image_r.cols;
		int rows = MAX(image_l.rows,image_r.rows);

		cv::Mat drawImage(rows, totalCols, image_l.type());
		
		if(drawImage.empty())
			return cv::Mat();

		cv::Mat submat = drawImage(cv::Rect(0,0,image_l.cols,image_l.rows));
		image_l.copyTo(submat);
		submat = drawImage(cv::Rect(image_l.cols,0,image_r.cols,image_r.rows));
		image_r.copyTo(submat);

		int max_index = _index_i > _index_j ? _index_i : _index_j;
		int min_index = _index_i < _index_j ? _index_i : _index_j;

		std::vector<cv::KeyPoint> keypoint_l = images_points[min_index];
		std::vector<cv::KeyPoint> keypoint_r = images_points[max_index];

		std::pair<int,int> index = std::make_pair(min_index,max_index);

		std::vector<cv::DMatch> matches = matches_matrix[index];

		for(int i=0; i < matches.size(); i++)
		{
			cv::Point2f pt1 = keypoint_l[matches[i].queryIdx].pt;
			cv::Point2f pt2 = keypoint_r[matches[i].trainIdx].pt;

			pt2.x += image_l.cols;

			cv::line(drawImage,pt1,pt2,cv::Scalar(0,0,255),1);
		}

		return drawImage;
	}

	CFeatureMatcher::~CFeatureMatcher()
	{
		delete m_sift;
	}
}

