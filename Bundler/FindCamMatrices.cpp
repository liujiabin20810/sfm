
#include "FindCamMatrices.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>

using namespace std;

namespace bundler{
	FindCamMatrices::FindCamMatrices(void)
	{
	}


	FindCamMatrices::~FindCamMatrices(void)
	{
	}


	bool FindCamMatrices::CheckCoherentRotation(cv::Mat_<double>& R) {
		if(fabsf(determinant(R))-1.0 > 1e-07) {
			cerr<<"det(R) != +-1.0, this is not a rotation matrix"<<endl;
			return false;
		}
		return true;
	}

	void FindCamMatrices::GetCameraMat(cv::Mat K,cv::Mat F,  cv::Matx34d &P)
	{
		//Essential matrix: compute then extract cameras [R|t]
		cv::Mat_<double> E = K.t() * F * K; //according to HZ (9.12)
		//decompose E to P' , HZ (9.19)
		cv::SVD svd(E,cv::SVD::MODIFY_A);
		cv::Mat svd_u = svd.u;
		cv::Mat svd_vt = svd.vt;
		cv::Mat svd_w = svd.w;
		cv::Matx33d W(0,-1,0,//HZ 9.13
			1,0,0,
			0,0,1);

		cv::Mat_<double> R = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
		cv::Mat_<double> t = svd_u.col(2); //u3
		if (!CheckCoherentRotation(R)) {
			cout<<"resulting rotation is not coherent\n";
			P = 0;
			return;
		}
		
		P = cv::Matx34d(R(0,0),R(0,1),R(0,2),t(0),
			R(1,0),R(1,1),R(1,2),t(1),
			R(2,0),R(2,1),R(2,2),t(2));
	
	
	}
	
	// AX = B , X = (x,y,z,1)
	cv::Mat_<double> FindCamMatrices::LinearLSTriangulation(
		cv::Point3d u,//homogenous image point (u,v,1)
		cv::Matx34d P,//camera 1 matrix
		cv::Point3d u1,//homogenous image point in 2nd camera
		cv::Matx34d P1//camera 2 matrix
		)
	{
		//build A matrix
		cv::Matx43d A(u.x*P(2,0)-P(0,0),u.x*P(2,1)-P(0,1),u.x*P(2,2)-P(0,2),
			u.y*P(2,0)-P(1,0),u.y*P(2,1)-P(1,1),u.y*P(2,2)-P(1,2),
			u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),u1.x*P1(2,2)-P1(0,2),
			u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),u1.y*P1(2,2)-P1(1,2)
			);
		//build B vector
		cv::Matx41d B(-(u.x*P(2,3)-P(0,3)),
			-(u.y*P(2,3)-P(1,3)),
			-(u1.x*P1(2,3)-P1(0,3)),
			-(u1.y*P1(2,3)-P1(1,3)));
		//solve for X
		cv::Mat_<double> X;
		cv::solve(A,B,X,cv::DECOMP_SVD);
		return X;
	}

	double FindCamMatrices::TriangulatePoints(vector<cv::Point2f> pt_set1,
		vector<cv::Point2f> pt_set2,
		const cv::Mat K,
		const cv::Mat Kinv,
		const cv::Matx34d& P,
		const cv::Matx34d& P1,
		vector<cv::Point3d>& pointcloud)
	{
		vector<double> reproj_error;
		int pts_size = pt_set1.size();
		for (unsigned int i=0; i<pts_size; i++) {
			//convert to normalized homogeneous coordinates
			cv::Point2f kp = pt_set1[i];
			cv::Point3d u(kp.x,kp.y,1.0);
			cv::Mat_<double> um = Kinv * (cv::Mat_<double>)(u);
			u = um.at<cv::Point3d>(0);

			cv::Point2f kp1 = pt_set2[i];
			cv::Point3d u1(kp1.x,kp1.y,1.0);
			cv::Mat_<double> um1 = Kinv * (cv::Mat_<double>)(u1);
			u1 = um1.at<cv::Point3d>(0);

			//triangulate
			cv::Mat_<double> X = LinearLSTriangulation(u,P,u1,P1);

			//calculate reprojection error
			cv::Mat_<double> xPt_img = K * cv::Mat(P1) * X;

			cv::Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

			reproj_error.push_back(norm(xPt_img_-kp1));

			//store 3D point
			pointcloud.push_back(cv::Point3d(X(0),X(1),X(2)));
		}
		//return mean reprojection error
		cv::Scalar me = cv::mean(reproj_error);

		return me[0];
	}
}
