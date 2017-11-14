#include <tchar.h>
#include <iostream>

#include <GL/glew.h>

#include "SIFTGPU.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace bundler

{
	cv::Mat mergeCols(cv::Mat A, cv::Mat B)
	{
		assert(A.type() == B.type());
		int totalCols = A.cols + B.cols;
		int rows = MAX(A.rows,B.rows);

		cv::Mat mergedDescriptors(rows, totalCols, A.type());
		
		if(mergedDescriptors.empty())
			return cv::Mat();

		cv::Mat submat = mergedDescriptors(cv::Rect(0,0,A.cols,A.rows));
		A.copyTo(submat);
		submat = mergedDescriptors(cv::Rect(A.cols,0,B.cols,B.rows));
		B.copyTo(submat);
		return mergedDescriptors;
	}

	cv::Mat mergeRows(cv::Mat A,cv::Mat B )
	{
		assert(A.type() == B.type());

		int totalRows = A.rows + B.rows;
		int cols = MAX(A.cols,B.cols);

		cv::Mat mergedDescriptors(totalRows, cols, A.type());

		if(mergedDescriptors.empty())
			return cv::Mat();
		
		cv::Mat submat = mergedDescriptors(cv::Rect(0,0,A.cols,A.rows));
		A.copyTo(submat);
		submat = mergedDescriptors(cv::Rect(0,A.rows,B.cols,B.rows));
		B.copyTo(submat);
		return mergedDescriptors;

	}

	SIFT::SIFT(int feature_num)
	{
		m_sift = new SiftGPU;
		
		int gpuType = m_sift->CreateContextGL();
		
		std::cout << gpuType<< std::endl;

		m_matcher = new SiftMatchGPU(feature_num);
		
		m_matcher->VerifyContextGL(); //must call once


	}
	
	SIFT::~SIFT()
	{
		if(m_sift)
			delete m_sift;
		
		if(m_matcher)
			delete m_matcher;
	}
		
	void SIFT::setImageList(std::vector<std::string> imgList)
	{
		m_nimage = imgList.size();

		for (int i =0; i< m_nimage && i < 50; i++)
		{
			int len = strlen(imgList[i].c_str());

			m_imgList[i] = imgList[i].c_str();
		}

		m_sift->SetImageList(m_nimage,m_imgList);

	}

	void SIFT::operator()(cv::Mat image, std::vector<SiftKeypoint>& keypoints) const
	{
		if(!m_sift->RunSIFT(image.cols,image.rows,image.data,GL_BGR,GL_UNSIGNED_BYTE))
			return;
		
		int N = m_sift->GetFeatureNum();
#ifdef _DEBUG
		std::cout<<"Num: "<<N<<std::endl;
#endif //_DEBUG
		

		keypoints.resize(N);
		
		m_sift->GetFeatureVector(&keypoints[0], NULL);
	}
	
	void SIFT::operator()(cv::Mat image, std::vector<SiftKeypoint>& keypoints, std::vector<float>& descriptors) const
	{
		if(!m_sift->RunSIFT(image.cols,image.rows,image.data,GL_BGR,GL_UNSIGNED_BYTE))
			return;
		
		int N = m_sift->GetFeatureNum();
//		std::cout<<"Num: "<<N<<std::endl;

		keypoints.resize(N);
		descriptors.resize(128*N);
		
		m_sift->GetFeatureVector(&keypoints[0], &descriptors[0]);

	}
	
	void SIFT::operator()(std::vector<std::vector<SiftKeypoint>>& keypoints, std::vector<std::vector<float>>&descriptors) const
	{
		keypoints.resize(m_nimage);
		descriptors.resize(m_nimage);
		
		for (int i=0; i< m_nimage; i++)
		{
			if(!m_sift->RunSIFT(i))
				continue;

			int N = m_sift->GetFeatureNum();
		
			keypoints[i].resize(N);
			descriptors[i].resize(128*N);

			m_sift->GetFeatureVector(&keypoints[i][0], &descriptors[i][0]);
		}
		
	}

	std::vector<int> SIFT::findInliers(std::vector<SiftGPU::SiftKeypoint> &queryKeypoints, std::vector<SiftGPU::SiftKeypoint> &objKeypoints,cv::Mat& F,int match_buf[][2], int match_num)
	{
		// º∆À„Fæÿ’Û
		std::vector<cv::Point2f> queryCoord;
		std::vector<cv::Point2f> objectCoord;

		for( int i = 0; i < match_num; i++){
			queryCoord.push_back(cv::Point2f(queryKeypoints[match_buf[i][0]].x ,queryKeypoints[match_buf[i][0]].y));
			objectCoord.push_back(cv::Point2f(objKeypoints[match_buf[i][1]].x ,objKeypoints[match_buf[i][1]].y));
		}

		cv::Mat mask;
		std::vector<cv::Point2f> queryInliers;
		std::vector<cv::Point2f> sceneInliers;
		
		F = cv::findFundamentalMat(queryCoord, objectCoord, mask, CV_FM_RANSAC);

		cv::vector<int> inliers;
		inliers.resize(match_num);

//		std::cout<<"∏ˆ ˝£∫ "<<mask.rows<<" "<<match_num<<" "<<mask.cols<<std::endl;

		int inliers_cnt = 0, outliers_cnt = 0;
		for (int j = 0; j < mask.rows; j++){
			if (mask.at<uchar>(j) == 1){
				inliers[j] = 1;
				inliers_cnt++;
			}else {
				inliers[j] = 0;
				outliers_cnt++;
			}
		}

#ifdef _DEBUG
		std::cout<<"Inliers Num: "<<inliers_cnt <<std::endl;
#endif
		
		return inliers;
	}
	
	std::vector<int> SIFT::match(std::vector<SiftGPU::SiftKeypoint> queryKeypoints, std::vector<SiftGPU::SiftKeypoint> trainedKeypoints,
		std::vector<float> queryDescriptors,std::vector<float> trainedDescriptors,int match_buffer[][2] )
	{
		int queryPtsNum = queryKeypoints.size();
		int trainedPtsNum = trainedKeypoints.size();

#ifdef _DEBUG
		std::cout<<"queryPtsNum = " << queryPtsNum << std::endl;
		std::cout<<"trainedPtsNum = " << trainedPtsNum << std::endl;
#endif // _DEBUG

		m_matcher->SetDescriptors(0, queryPtsNum, &queryDescriptors[0]); //image 1
		m_matcher->SetDescriptors(1, trainedPtsNum, &trainedDescriptors[0]); //image 2

		int (*_buf)[2] = new int[queryPtsNum][2];
		int num_match = m_matcher->GetSiftMatch(queryPtsNum, _buf);

#ifdef _DEBUG
		std::cout << num_match << " sift matches were found;\n";
#endif // _DEBUG

		std::vector<int> mask;
		int good_match_num = 0;
		int j = 0;

		if(num_match > 16)
		{
			cv::Mat F;
			mask = findInliers(queryKeypoints,trainedKeypoints,F,_buf,num_match);
			
			for (int i=0; i< num_match; i++)
			{
				if(mask[i])
				{
					match_buffer[j][0] = _buf[i][0];
					match_buffer[j][1] = _buf[i][1];

					j++;
				}
			}

			good_match_num = j;
		}

#ifdef _DEBUG		
		std::cout<<"good match num : "<<good_match_num << std::endl;
#endif // _DEBUG

		delete[] _buf;

		return mask;
	}

	int SIFT::match(std::vector<SiftGPU::SiftKeypoint> queryKeypoints, std::vector<SiftGPU::SiftKeypoint> trainedKeypoints,
		std::vector<float> queryDescriptors, std::vector<float> trainedDescriptors, int match_buffer[][2], cv::Mat &F)
	{
		int queryPtsNum = queryKeypoints.size();
		int trainedPtsNum = trainedKeypoints.size();

#ifdef _DEBUG
		std::cout << "queryPtsNum = " << queryPtsNum << std::endl;
		std::cout << "trainedPtsNum = " << trainedPtsNum << std::endl;
#endif // _DEBUG

		m_matcher->SetDescriptors(0, queryPtsNum, &queryDescriptors[0]); //image 1
		m_matcher->SetDescriptors(1, trainedPtsNum, &trainedDescriptors[0]); //image 2

		int(*_buf)[2] = new int[queryPtsNum][2];
		int num_match = m_matcher->GetSiftMatch(queryPtsNum, _buf);

#ifdef _DEBUG
		std::cout << num_match << " sift matches were found;\n";
#endif // _DEBUG

		std::vector<int> mask;
		int good_match_num = 0;
		int j = 0;

		if ( num_match > 50 )
		{
			mask = findInliers(queryKeypoints, trainedKeypoints, F, _buf, num_match);

			for (int i = 0; i< num_match; i++)
			{
				if (mask[i])
				{
					match_buffer[j][0] = _buf[i][0];
					match_buffer[j][1] = _buf[i][1];

					j++;
				}
			}

			good_match_num = j;
		}

#ifdef _DEBUG		
		std::cout << "good match num : " << good_match_num << std::endl;
#endif // _DEBUG

		delete[] _buf;

		return good_match_num;
	}

	std::vector<int> SIFT::match(std::vector<std::vector<SiftKeypoint>> Keypoints,std::vector<std::vector<float>> Descriptors,
		std::vector<std::vector<std::pair<int ,int >>>& match_buffer)
	{
		std::vector<int> good_match_num;
		good_match_num.resize(m_nimage);

		match_buffer.clear();

		std::vector<SiftKeypoint> queryKpts,trainedKpts;
		std::vector<float> queryDes,trainedDes;

		for (int i=0; i < m_nimage-1; i++)
		{
			queryKpts = Keypoints[i];
			queryDes = Descriptors[i];

			int n = queryKpts.size();
			int (*_buf)[2] = new int[n][2];

			for(int j = i+1; j< m_nimage; j++)
			{
				std::cout<<i<<" "<< j <<": \n";

				trainedKpts = Keypoints[j];
				trainedDes = Descriptors[j];

				std::vector<int> match_mask = match(queryKpts, trainedKpts, queryDes, trainedDes, _buf);

				std::vector<std::pair<int,int>> match_buf;

				int _index = 0;
				for (int k = 0; k< match_mask.size(); k++)
				{
					if (match_mask[k] == 0)
						continue;

					int queryIdx = _buf[_index][0];
					int trainedIdx = _buf[_index][1];
					match_buf.push_back(std::make_pair(queryIdx,trainedIdx));

					_index++;
				}
				std::cout << std::endl;

				match_buffer.push_back(match_buf);

			}

			delete [] _buf;
		}

		return good_match_num;
	}

	void SIFT::writeSiftMatch(std::ofstream &fp, std::vector<std::vector<std::pair<int ,int >>>& match_buffer)
	{
		int idx = 0;
		for(int i = 0; i < m_nimage-1; i++)
			for(int j = i + 1; j < m_nimage; j++)
			{
				std::vector<std::pair<int ,int >> buff = match_buffer[idx];
				idx++;

				int match_num = buff.size();

				fp << i <<" "<<j<<" "<<match_num<<std::endl;

				for (int k = 0; k < match_num; k++)
				{
					fp<<buff[k].first<<" "<<buff[k].second<<std::endl;
				}
			}
	}

	void SIFT::readSiftMatch(std::ifstream &fin, std::vector<std::vector<std::pair<int ,int >>>& match_buffer)
	{
		int i,j,match_num;

		while( fin >> i >> j >> match_num )
		{
			std::cout<<i<<" "<<j<<" "<<match_num<<std::endl;
			int i_idx, j_idx;

			std::vector<std::pair<int ,int >> buff;
//			buff.resize(match_num);

			for(int k=0; k<match_num; k++)
			{
				fin >> i_idx >> j_idx;
				std::cout<< i_idx <<" "<<j_idx<<std::endl;

				buff.push_back(std::make_pair(i_idx,j_idx));
			}

			match_buffer.push_back(buff);
		}
	}

	void SIFT::drawSiftMatch(cv::Mat src1,std::vector<SiftGPU::SiftKeypoint> keys1, cv::Mat src2, std::vector<SiftGPU::SiftKeypoint> keys2, int match_buf[][2] , int num_match)
	{

		std::vector<cv::Point2f> points1, points2;
		cv::Mat img_match;

		bool flag = true;

		if(flag)
			img_match = mergeCols(src1,src2);
		else
			img_match = mergeRows(src1,src2);
		
	//	std::cout << img_match.cols << " "<< img_match.rows<< std::endl;

		if(img_match.empty())
		{
			std::cout<<"merge image err..."<<std::endl;
			return;

		}

		//enumerate all the feature matches
		for(int i  = 0; i < num_match; ++i)
		{
			
			//How to get the feature matches: 
			SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];

			SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];

//			points1.push_back(Point2f(key1.x,key1.y));		
//			if(flag)
//				points2.push_back(Point2f(src1.cols + key2.x,key2.y));	
//			else
//				points2.push_back(Point2f(key2.x,src1.rows +key2.y));
//			if(inliers[i])
			{
				if(flag)
					cv::line(img_match,cv::Point2f(key1.x,key1.y),cv::Point2f(src1.cols +key2.x,key2.y),cv::Scalar(0,0,255),1);
				else
					cv::line(img_match,cv::Point2f(key1.x,key1.y),cv::Point2f(key2.x,src1.rows + key2.y),cv::Scalar(0,0,255),1);
			}
// 			else
// 			{
// 				if(flag)
// 					cv::line(img_match,cv::Point2f(key1.x,key1.y),cv::Point2f(src1.cols +key2.x,key2.y),cv::Scalar(255,0,0),1);
// 				else
// 					cv::line(img_match,cv::Point2f(key1.x,key1.y),cv::Point2f(src1.cols +key2.x,key2.y),cv::Scalar(255,0,0),1);
// 			}
		}

		cv::imwrite("match.jpg",img_match);
		cv::namedWindow("Match",1);
		cv::imshow("Match",img_match);
	 
		cv::waitKey(0);

	}

	void SIFT::ParseParam(int argc, char **argv)
	{
		//	 char * argv[] = {"-fo", "-1",  "-v", "1"};//
		//	 int argc = sizeof(argv)/sizeof(char*);
		
		m_sift->ParseParam(argc, argv);
	}
	
	void SIFT::convertSiftKeypoint(std::vector<SiftKeypoint> keys, std::vector<cv::Point2f>& points)
	{
		int keysNum = keys.size();

		points.clear();
		points.resize(keysNum);

		for (int i=0; i<keysNum; i++)
		{
			SiftKeypoint siftkey = keys[i];

			points[i] = cv::Point2f(siftkey.x,siftkey.y);
		}
	}
	
	void SIFT::convertDescriptor(std::vector<float> vdescriptor, cv::Mat & mdescriptor)
	{
		int pts_num = int(vdescriptor.size())/128;

		//cout<<" pts_num: "<<pts_num<<endl;
		if(mdescriptor.empty())
			mdescriptor = cv::Mat(pts_num,128,CV_32F);

		memcpy(mdescriptor.data, &vdescriptor[0], vdescriptor.size()*sizeof(float));

	}
}