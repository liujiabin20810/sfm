#ifndef  BUNDLER_SIFTGPU_HPP__
#define BUNDLER_SIFTGPU_HPP__
#pragma  once

#pragma warning(disable: 4244 18 4996 4800)

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <string>
#include <fstream>

#include "SiftGPU.h"

namespace bundler
{
	class SIFT
	{
		public:
		
			SIFT(int feature_num);
			
			~SIFT();
			
			void ParseParam(int argc, char **argv);

			void setImageList(std::vector<std::string> imgList);

			void operator()(cv::Mat image, std::vector<SiftKeypoint>& keypoints) const;
			
			void operator()(cv::Mat image, std::vector<SiftKeypoint>& keypoints, std::vector<float>& descriptors) const;

			void operator()(std::vector<std::vector<SiftKeypoint>>& keypoints, std::vector<std::vector<float>>& descriptors) const;

			std::vector<int> match(std::vector<SiftGPU::SiftKeypoint> queryKeypoints, std::vector<SiftGPU::SiftKeypoint> trainedKeypoints,
				std::vector<float> queryDescriptors,std::vector<float> trainedDescriptors,int match_buffer[][2] );

			int match(std::vector<SiftGPU::SiftKeypoint> queryKeypoints, std::vector<SiftGPU::SiftKeypoint> trainedKeypoints,
				std::vector<float> queryDescriptors, std::vector<float> trainedDescriptors, int match_buffer[][2],cv::Mat &F);

			std::vector<int> match(std::vector<std::vector<SiftKeypoint>> Keypoints,std::vector<std::vector<float>> Descriptors,
				std::vector<std::vector<std::pair<int ,int >>>& match_buffer);


			void drawSiftMatch(cv::Mat src1,std::vector<SiftGPU::SiftKeypoint> keys1, cv::Mat src2, std::vector<SiftGPU::SiftKeypoint> keys2, 
			int match_buf[][2] , int num_match);

			void writeSiftMatch(std::ofstream &fp, std::vector<std::vector<std::pair<int ,int >>>& match_buffer);
			
			void readSiftMatch(std::ifstream &fp, std::vector<std::vector<std::pair<int ,int >>>& match_buffer);
			
			void convertSiftKeypoint(std::vector<SiftKeypoint> keys, std::vector<cv::Point2f>& points);
			
			void convertSiftKeypoint(std::vector<SiftKeypoint> keys, std::vector<cv::KeyPoint>& points);

			void convertDescriptor(std::vector<float> vdescriptor, cv::Mat & mdescriptor);
			
			std::vector<int> findInliers(std::vector<SiftGPU::SiftKeypoint> &queryKeypoints, std::vector<SiftGPU::SiftKeypoint> &objKeypoints,
				cv::Mat& F,int match_buf[][2], int match_num);

		public:
			SiftGPU *m_sift;
			SiftMatchGPU *m_matcher;

			int m_nimage ;
			const char* m_imgList[50];
		
	};
}

#endif