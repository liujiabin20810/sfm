#ifndef  BUNDLER_FEATUREMATCHER_HPP__
#define BUNDLER_FEATUREMATCHER_HPP__
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <map>
#include <string>
#include <fstream>

namespace bundler
{
	class SIFT;

	class CPruneFeature
	{
	public:
		CPruneFeature(std::vector<std::string> feature_list, std::vector<cv::Mat> mask_list );
		
		~CPruneFeature();

		bool read_sift_features();

		cv::Mat drawImageMatches(int _index_i, int _index_j);

	public:
		//images num
		int n_images;
		// object mask
		std::vector<cv::Mat> m_image_masks;
		//prune matches with object mask
		std::map<std::pair<int, int>, std::vector<cv::DMatch> > prune_matches_matrix;
		// prune keypoints with object mask
		std::vector<std::vector<cv::KeyPoint>> prune_images_points;

		std::map<std::pair<int, int>, cv::Mat > prune_F_matrix;

		std::vector<std::string> m_feature_list;  //image name used to get sift feature name

	private:

		SIFT* m_sift;
		

	};

	class CFeatureMatcher
	{
	public:
		CFeatureMatcher();

		CFeatureMatcher(std::vector<std::string> images_list,std::vector<cv::Mat> images);

		void init_images_list(std::vector<std::string> images_list);

		void init_image_mask(std::vector<cv::Mat> image_masks);

		// init with images and calculate sift feature
		bool calc_sift_feature(std::vector<cv::Mat> images);

		// init with images list and calculate sift feature
		bool calc_sift_feature_and_match();

		bool read_sift_feature();

		bool calc_sift_feature();

		~CFeatureMatcher();

		cv::Mat drawImageMatches(int _index_i, int _index_j);

	private:

		std::vector<std::vector<float>> vec_descriptors;

		bool m_init_images_list;

		SIFT* m_sift;

	public:

		// good matches based on F matrix
		std::map<std::pair<int, int>, std::vector<cv::DMatch> > matches_matrix;

		std::map<std::pair<int, int>, cv::Mat > F_matrix;

		// source images Keypoints
		std::vector<std::vector<cv::KeyPoint>> images_points;

		std::vector<std::vector<cv::Vec3b> > images_points_colors;
		//images num
		int n_images;

		std::vector<std::string> m_images_list;

	};
}

#endif


