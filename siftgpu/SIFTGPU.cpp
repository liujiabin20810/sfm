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
		
		m_good_match_threshold = 8;

		int gpuType = m_sift->CreateContextGL();
		
		//std::cout << gpuType<< std::endl;

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
		m_strimgList.resize(m_nimage);
		for (int i =0; i< m_nimage && i < 50; i++)
		{
			int len = strlen(imgList[i].c_str());

			m_imgList[i] = imgList[i].c_str();
			m_strimgList[i] = imgList[i];
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

#define Vx(v) (v)[0]
#define Vy(v) (v)[1]
#define Vz(v) (v)[2]
	
	double fmatrix_compute_residual(double *F, double* r, double* l)
	{
		double Fl[3], Fr[3], pt;    

		Fl[0] = F[0] * Vx(l) + F[1] * Vy(l) + F[2] * Vz(l);
		Fl[1] = F[3] * Vx(l) + F[4] * Vy(l) + F[5] * Vz(l);
		Fl[2] = F[6] * Vx(l) + F[7] * Vy(l) + F[8] * Vz(l);

		Fr[0] = F[0] * Vx(r) + F[3] * Vy(r) + F[6] * Vz(r);
		Fr[1] = F[1] * Vx(r) + F[4] * Vy(r) + F[7] * Vz(r);
		Fr[2] = F[2] * Vx(r) + F[5] * Vy(r) + F[8] * Vz(r);

		pt = Vx(r) * Fl[0] + Vy(r) * Fl[1] + Vz(r) * Fl[2];


		return
			(1.0 / (Fl[0] * Fl[0] + Fl[1] * Fl[1] + 1e-8) +
			1.0 / (Fr[0] * Fr[0] + Fr[1] * Fr[1] + 1e-8 )) *
			(pt * pt);
	}

	std::vector<int> SIFT::findInliers(std::vector<SiftGPU::SiftKeypoint> &queryKeypoints, std::vector<SiftGPU::SiftKeypoint> &objKeypoints,cv::Mat& F,int match_buf[][2], int match_num)
	{
		// 计算F矩阵
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

		double f[9];
		f[0] = F.ptr<double>(0)[0];
		f[1] = F.ptr<double>(0)[1];
		f[2] = F.ptr<double>(0)[2];
		f[3] = F.ptr<double>(1)[0];
		f[4] = F.ptr<double>(1)[1];
		f[5] = F.ptr<double>(1)[2];
		f[6] = F.ptr<double>(2)[0];
		f[7] = F.ptr<double>(2)[1];
		f[8] = F.ptr<double>(2)[2];

		cv::vector<int> inliers;
		inliers.resize(match_num);

		//std::cout<<"个数： "<<mask.rows<<" "<<match_num<<" "<<f[0]<<std::endl;
		std::vector<double> residuals(match_num);
		
		double threshold = 5.0625;
		int inliers_cnt = 0, outliers_cnt = 0;
		double sum_residual = 0.0;

		std::vector<int> preIndex;
		for (int j = 0; j < mask.rows; j++)
		{
			if (mask.at<uchar>(j) == 1)
			{
				
				double lp[3] = {0.0,0.0,1.0},rp[3] = {0.0,0.0,1.0};
				lp[0] = queryCoord[j].x;
				lp[1] = queryCoord[j].y;

				rp[0] = objectCoord[j].x;
				rp[1] = objectCoord[j].y;
				
				double dist = fmatrix_compute_residual(f,rp,lp);
				
				if(threshold < dist)
				{
					inliers[j] = 0;
				}
				else
				{
					inliers[j] = 1;
					inliers_cnt++;
					preIndex.push_back(j);

					sum_residual += dist;
				}
			}
			else
			{
				inliers[j] = 0;
			}
		}		

		//std::cout<<"[F matrix] first inliers: "<< inliers_cnt <<std::endl;		
		if(inliers_cnt < m_good_match_threshold)
			return inliers;

		queryCoord.clear();
		objectCoord.clear();
		for( int i = 0; i < match_num; i++){
			if(!inliers[i])
				continue;
			queryCoord.push_back(cv::Point2f(queryKeypoints[match_buf[i][0]].x ,queryKeypoints[match_buf[i][0]].y));
			objectCoord.push_back(cv::Point2f(objKeypoints[match_buf[i][1]].x ,objKeypoints[match_buf[i][1]].y));
		}
		
		cv::Mat mask2;
		F = cv::findFundamentalMat(queryCoord, objectCoord, mask2, CV_FM_RANSAC);

		f[0] = F.ptr<double>(0)[0];
		f[1] = F.ptr<double>(0)[1];
		f[2] = F.ptr<double>(0)[2];
		f[3] = F.ptr<double>(1)[0];
		f[4] = F.ptr<double>(1)[1];
		f[5] = F.ptr<double>(1)[2];
		f[6] = F.ptr<double>(2)[0];
		f[7] = F.ptr<double>(2)[1];
		f[8] = F.ptr<double>(2)[2];

		inliers_cnt = 0;
		sum_residual = 0.0;
		int count_residual = 0;
		for (int j = 0; j < mask2.rows; j++){
			if (mask2.at<uchar>(j) == 1)
			{

				double lp[3] = {0.0,0.0,1.0},rp[3] = {0.0,0.0,1.0};
				lp[0] = queryCoord[j].x;
				lp[1] = queryCoord[j].y;

				rp[0] = objectCoord[j].x;
				rp[1] = objectCoord[j].y;

				double dist = fmatrix_compute_residual(f,rp,lp);
				if(dist > threshold)
				{
					count_residual++;
					int preidx = preIndex[j];
					inliers[preidx] = 0;
				}
				else
				{
					sum_residual += dist;
					inliers_cnt++;
				}
			}
			else
			{
				count_residual++;
				int preidx = preIndex[j];
				inliers[preidx] = 0;
			}
		}

#ifdef _DEBUG
		std::cout<<"inliers num: "<<inliers_cnt <<std::endl;
#endif
//		if(inliers_cnt > m_good_match_threshold )
//			std::cout<<"inliers num: "<<inliers_cnt<<" re-project err: "<< sum_residual/inliers_cnt <<std::endl;
		
		return inliers;
	}
	
// 	std::vector<int> SIFT::match(std::vector<SiftGPU::SiftKeypoint> queryKeypoints, std::vector<SiftGPU::SiftKeypoint> trainedKeypoints,
// 		std::vector<float> queryDescriptors,std::vector<float> trainedDescriptors,int match_buffer[][2] )
// 	{
// 		int queryPtsNum = queryKeypoints.size();
// 		int trainedPtsNum = trainedKeypoints.size();
// 
// #ifdef _DEBUG
// 		std::cout<<"queryPtsNum = " << queryPtsNum << std::endl;
// 		std::cout<<"trainedPtsNum = " << trainedPtsNum << std::endl;
// #endif // _DEBUG
// 
// 		m_matcher->SetDescriptors(0, queryPtsNum, &queryDescriptors[0]); //image 1
//		m_matcher->SetDescriptors(1, trainedPtsNum, &trainedDescriptors[0]); //image 2
// 
// 		int (*_buf)[2] = new int[queryPtsNum][2];
// 		int num_match = m_matcher->GetSiftMatch(queryPtsNum, _buf);
// 
// #ifdef _DEBUG
// 		std::cout << num_match << " sift matches were found;\n";
// #endif // _DEBUG
// 
// 		std::vector<int> mask;
// 		int good_match_num = 0;
// 		int j = 0;
// 
// 		if(num_match > 16)
// 		{
// 			cv::Mat F;
// 			mask = findInliers(queryKeypoints,trainedKeypoints,F,_buf,num_match);
// 			
// 			for (int i=0; i< num_match; i++)
// 			{
// 				if(mask[i])
// 				{
// 					match_buffer[j][0] = _buf[i][0];
// 					match_buffer[j][1] = _buf[i][1];
// 
// 					j++;
// 				}
// 			}

//			good_match_num = j;
//		}
// 
// #ifdef _DEBUG		
// 		std::cout<<"good match num : "<<good_match_num << std::endl;
// #endif // _DEBUG
// 
// 		delete[] _buf;
// 
// 		return mask;
// 	}

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

		if ( num_match > m_good_match_threshold )
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

			good_match_num = j  >= m_good_match_threshold ? j : 0 ;
		}

#ifdef _DEBUG		
		std::cout << "good match num : " << good_match_num << std::endl;
#endif // _DEBUG

		delete[] _buf;

		return good_match_num ;
	}

// 	std::vector<int> SIFT::match(std::vector<std::vector<SiftKeypoint>> Keypoints,std::vector<std::vector<float>> Descriptors,
// 		std::vector<std::vector<std::pair<int ,int >>>& match_buffer)
// 	{
// 		std::vector<int> good_match_num;
// 		good_match_num.resize(m_nimage);
// 
// 		match_buffer.clear();
// 
// 		std::vector<SiftKeypoint> queryKpts,trainedKpts;
// 		std::vector<float> queryDes,trainedDes;
// 
// 		for (int i=0; i < m_nimage-1; i++)
// 		{
// 			queryKpts = Keypoints[i];
// 			queryDes = Descriptors[i];
// 
// 			int n = queryKpts.size();
// 			int (*_buf)[2] = new int[n][2];
// 
// 			for(int j = i+1; j< m_nimage; j++)
// 			{
// 				std::cout<<i<<" "<< j <<": \n";
// 
// 				trainedKpts = Keypoints[j];
// 				trainedDes = Descriptors[j];
// 
// 				std::vector<int> match_mask = match(queryKpts, trainedKpts, queryDes, trainedDes, _buf);
// 
// 				std::vector<std::pair<int,int>> match_buf;
// 
// 				int _index = 0;
// 				for (int k = 0; k< match_mask.size(); k++)
// 				{
// 					if (match_mask[k] == 0)
// 						continue;
// 
// 					int queryIdx = _buf[_index][0];
// 					int trainedIdx = _buf[_index][1];
// 					match_buf.push_back(std::make_pair(queryIdx,trainedIdx));
// 
// 					_index++;
// 				}
// 				std::cout << std::endl;
// 
// 				match_buffer.push_back(match_buf);
// 
// 			}
// 
// 			delete [] _buf;
// 		}
// 
// 		return good_match_num;
// 	}

	bool SIFT::writeSiftMatch(std::ofstream &fp, std::vector<cv::DMatch> & matches)
	{
		if(fp.is_open())
		{
			int match_num = matches.size();

			for (int k = 0; k < match_num; k++)
				fp<<matches[k].queryIdx<<" ";
			fp<<std::endl;
			for (int k = 0; k < match_num; k++)
				fp<<matches[k].trainIdx<<" ";
			fp<<std::endl;
			return true;
		}
		else
			return false;
	}

	bool SIFT::writeSiftMatch( std::map<std::pair<int, int>,std::vector<cv::DMatch> >& matches_matrix)
	{
		int i,j;
		int idx = 0;

		//create .mat file 
		std::string filename = m_strimgList[0];
		int pos = filename.rfind("/");
		if(pos < 0 )
			return false;
		std::string outname = filename.substr(0,pos) + "/full_match.txt";
		std::ofstream fout(outname);
		if(!fout.is_open())
			return false;

		for(i = 0 ; i < m_nimage-1; i++)
		{
			for (j = i+1 ; j < m_nimage; j++)
			{
				std::vector<cv::DMatch> _matches = matches_matrix[std::make_pair(i,j)];

				if(_matches.empty())
					continue;

				pos = m_strimgList[i].rfind("/");
				std::string img1,img2;
				if(pos > 0)
				{
					img1 = m_strimgList[i].substr(pos+1);
					img2 = m_strimgList[j].substr(pos+1);
					fout <<img1<<" "<<img2<<" "<<_matches.size()<<std::endl;

					//std::cout<<img1<<" "<<img2<<std::endl;
					writeSiftMatch(fout,_matches);
				}
			}

		}

		fout.close();
		return true;
	}
	
	//Read the .sift files in C.WU's BINARY format
	bool SIFT::readSiftFeature(std::ifstream &fin , std::vector<SiftGPU::SiftKeypoint>& keys,std::vector<float>& descs)
	{
		char ch[256];
		
		fin.read(ch,sizeof(int)*5);
		char name[5]={};
		memcpy(name,ch,4);
		
		char version[5] = {};
		memcpy(version,ch+4,4);

		int npoint = 0 , n = 0 , dim = 0;
		memcpy(&npoint, ch+8, 4);
		memcpy(&n, ch+12, 4);
		memcpy(&dim, ch+16, 4);
//		npoint = (ch[11]<<24) + (ch[10]<<16) + (ch[9]<<8) + ch[8];
//		n = (ch[15]<<24) + (ch[14]<<16) + (ch[13]<<8) + ch[12];
//		dim =  ((ch[19]<<24) + (ch[18]<<16) + (ch[17]<<8) + ch[16]);

		if(n != 5 || dim != 128 || (strcmp(name,"SIFT") != 0)  || (strcmp(version,"V4.0") != 0 ))
		{
			printf("%s,%s,%d,%d,%d\n",name,version,npoint,n,dim);
			return false;
		}

		keys.resize(npoint);
		descs.resize(npoint*dim);
		//keyPoint
		for (int i=0; i < npoint; i++)
		{
			fin.read(ch,sizeof(float)*5);
			float x,y,scale,orientation;
			unsigned char color[4];
			memcpy(&x,ch,4);
			memcpy(&y,ch+4,4);
			memcpy(color,ch+8,4);
			memcpy(&scale,ch+12,4);
			memcpy(&orientation,ch+16,4);
// 			if(i < 5 )
// 			{
// 				printf("%d,%d,%d",int(color[0]),int(color[1]),int(color[2]));
// 				printf("\t%f,\t%f,\t%f,\t%f\n",x,y,scale,orientation);
// 			}
			keys[i].x = x;
			keys[i].y = y;
			keys[i].s = scale;
			keys[i].o = orientation;
		}

		std::vector<float>::iterator iter = descs.begin();
		
		for(int i =0; i < npoint && iter != descs.end(); i++)
		{			
			fin.read(ch,dim);

			std::vector<float> _feature(dim);
			double _fsum = 0;
			for(int j = 0; j < dim; j++)
			{
				char c = (uchar)ch[j];
				_feature[j] = (float)c;
				_fsum += _feature[j]*_feature[j];
			}
			_fsum = sqrtf(_fsum);
 			//printf("%f \n",_fsum);

			for(int j = 0; j < dim && iter != descs.end(); j++)
			{
				_feature[j] = _feature[j]/_fsum;  // 归一化
				*iter = _feature[j];
				++iter;
				//printf("%d, ",(int)_feature[j]);
			}
			//printf("\n");
		}

		fin.read(ch,sizeof(int));
		int get_eof;
		memcpy(&get_eof,ch,4);
		int eof_marker = (0xff+('E'<<8)+('O'<<16)+('F'<<24));

		if(get_eof != eof_marker)
			return false;

		return true;
	}
	//Write the .sift files in C.WU's BINARY format
	bool SIFT::writeSiftFeature(std::ofstream &fp , std::vector<SiftGPU::SiftKeypoint> keys,std::vector<float> descs)
	{
		char name[5] = "SIFT";
		char version[5] = "V4.0";
		//int name = (int)('S'+ ('I'<<8)+('F'<<16)+('T'<<24));
		//int version = (int)('V'+('4'<<8)+('.'<<16)+('0'<<24));
		//(int)(0xff+('E'<<8)+('O'<<16)+('F'<<24));

		int eof_marker = (0xff+('E'<<8)+('O'<<16)+('F'<<24));
		int npoint = keys.size();
		int n = 5,dim = 128;
		char ch[256];
		memcpy(ch,name,4);
		memcpy(ch+4,version,4);
		memcpy(ch+8,&npoint,4);
		memcpy(ch+12,&n,4);
		memcpy(ch+16,&dim,4);

		fp.write(ch,4*5);

		for (int i = 0; i < npoint; i++)
		{
			SiftGPU::SiftKeypoint p = keys[i];

			//std::cout<<p.x<<" "<<p.y<<" "<<p.s<<" "<<p.o<<std::endl;
			unsigned char color[4] = {255,255,255};
			char ch[256];
			memcpy(ch,&p.x,4);
			memcpy(ch+4,&p.y,4);
			memcpy(ch+8,color,4);
			memcpy(ch+12,&p.s,4);
			memcpy(ch+16,&p.o,4);

			fp.write(ch,sizeof(float)*5);
		}

		int step = 0;
		int maxData = 0;
		int minData = 127;
		for (int i = 0; i < npoint; i++)
		{
			std::vector<float> _feature(dim);
			double _fsum = 0.0;
			for (int j=0; j < dim; j++)
			{
				int position = i*dim + j;
				float f = descs[position];
				_feature[j] = f;
				//_fsum += f*f;
			}

			char *_chfeature = new char[dim];

			for (int j=0; j < dim; j++)
			{
				_feature[j] = _feature[j]*512;
				int tmp = (int)(_feature[j] + 0.5) ;
				_chfeature[j] = (char)tmp;
				//if(tmp > maxData) maxData = tmp;
				//if(tmp < minData) minData = tmp;
				//printf("%d, ",tmp);
			}			
			
			fp.write(_chfeature,sizeof(char)*dim);

			delete [] _chfeature;
		}

		memcpy(ch,&eof_marker,4);
		fp.write(ch,4);
		//printf("descriptor range: %d,%d\n",maxData,minData);
		return true;
	}	
	//Write the .sift files in C.WU's BINARY format
	bool SIFT::writeSiftFeature(std::vector<std::vector<SiftGPU::SiftKeypoint> > keys,std::vector<std::vector<float> > descs)
	{
		assert(keys.size() == m_nimage && keys.size() == descs.size());
		
		for(int i=0; i < m_nimage;i++)	
		{
			std::string filename = m_strimgList[i];
			int pos = filename.rfind(".");
			if(pos < 0)
				return false;
			std::string outname = filename.substr(0,pos) + ".sift";
			//std::cout<<filename<<" "<<pos<<" "<<outname<<std::endl;
			std::ofstream fp(outname,std::ios_base::binary);
			if(!fp.is_open())
				return false;
			
			writeSiftFeature(fp,keys[i],descs[i]);

			fp.close();
		}

		return true;
	}
	//Write the .sift files in Lowe's ASCII format
	bool SIFT::writeLoweSiftFeature(std::ofstream &fp , std::vector<SiftGPU::SiftKeypoint> keys,std::vector<float > descs)
	{
		int npoint = keys.size();
		int dim = 128;

		int ndes = descs.size();
		assert(ndes == dim*npoint);

		fp << npoint<<" "<< dim <<std::endl;
		// write echo keypoint feature;
		// y,x,scale,orientation;
		// 128-d descriptors , 20s each line

		for (int i = 0 ; i < npoint; ++i)
		{
			SiftGPU::SiftKeypoint p = keys[i];
			fp << (p.y -0.5)<<" "<<(p.x - 0.5 )<< " "<<p.s<<" "<< -p.o<<std::endl;

			std::vector<float> _descriptors(dim);
			double _dsum = 0.0;
			for (int j= 0; j < dim; j++)
			{
				int position = i*dim + j;
				_descriptors[j] = descs[position];

				_dsum += descs[position]*descs[position];
			}

			_dsum = sqrt(_dsum);

			for (int j= 0; j < dim; j++)
			{
				int _descriptor = (int)(512*_descriptors[j]/_dsum);
				if(j != 0 && j % 20 == 0)
					fp<<std::endl;
				
				fp <<" "<<_descriptor;
			}

			fp<<std::endl;
		}
		return true;
	}
	//Write the .sift files in Lowe's ASCII format
	bool SIFT::writeLoweSiftFeature(std::vector<std::vector<SiftGPU::SiftKeypoint> > keys,std::vector<std::vector<float> > descs)
	{
		assert(keys.size() == m_nimage && keys.size() == descs.size());

		for(int i=0; i < m_nimage;i++)	
		{
			std::string filename = m_strimgList[i];
			int pos = filename.rfind(".");
			if(pos < 0)
				return false;
			std::string outname = filename.substr(0,pos) + ".sift";
			//std::cout<<filename<<" "<<pos<<" "<<outname<<std::endl;
			std::ofstream fp(outname);
			if(!fp.is_open())
				return false;

			writeLoweSiftFeature(fp,keys[i],descs[i]);

			fp.close();
		}
		return true;
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
		cv::namedWindow("Match",0);
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