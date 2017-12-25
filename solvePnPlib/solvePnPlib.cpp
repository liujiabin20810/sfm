// solvePnPlib.cpp : 定义 DLL 应用程序的导出函数。
//
#pragma  once
#include "solvePnPlib.h"
#include "triangulate.h"
#include "matrix.h"
#include "keys.h"
#include "5point.h"
#include "Epipolar.h"
#include "defines.h"

#include <iostream>
#include <fstream>

#ifdef _DEBUG
#pragma comment(lib,"opencv_core249d.lib")
#pragma comment(lib,"opencv_imgprc249d.lib")
#pragma comment(lib,"opencv_calib3d249d.lib")

#else
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"cminpack.lib")
#pragma comment(lib,"matrix.lib")
#pragma comment(lib,"cblas.lib")
#pragma comment(lib,"clapack.lib")
#pragma comment(lib,"f2c.lib")
#endif // _DEBUG


/* Use a 180 rotation to fix up the intrinsic matrix */
void FixIntrinsics(double *P, double *K, double *R, double *t) 
{
	/* Check the parity along the diagonal */
	int neg = (K[0] < 0.0) + (K[4] < 0.0) + (K[8] < 0.0);

	/* If odd parity, negate the instrinsic matrix */
	if ((neg % 2) == 1) {
		matrix_scale(3, 3, K, -1.0, K);
		matrix_scale(3, 4, P, -1.0, P);
	}

	/* Now deal with case of even parity */
	double fix[9];
	matrix_ident(3, fix);
	double tmp[9], tmp2[12];

	if (K[0] < 0.0 && K[4] < 0.0) {
		fix[0] = -1.0;
		fix[4] = -1.0;
	} else if (K[0] < 0.0) {
		fix[0] = -1.0;
		fix[8] = -1.0;
	} else if (K[4] < 0.0) {
		fix[4] = -1.0;
		fix[8] = -1.0;
	} else {
		/* No change needed */
	}

	matrix_product(3, 3, 3, 3, K, fix, tmp);
	memcpy(K, tmp, sizeof(double) * 3 * 3);

	double Kinv[9];
	matrix_invert(3, K, Kinv);

	matrix_product(3, 3, 3, 4, Kinv, P, tmp2);

	memcpy(R + 0, tmp2 + 0, sizeof(double) * 3);
	memcpy(R + 3, tmp2 + 4, sizeof(double) * 3);
	memcpy(R + 6, tmp2 + 8, sizeof(double) * 3);

	t[0] = tmp2[3];
	t[1] = tmp2[7];
	t[2] = tmp2[11];
}

SOLVEPNPLIB_API bool FindAndVerifyCamera(int num_points, 
	std::vector<cv::Point3f> points3d,
	std::vector<cv::Point2f> points2d,
	std::vector<int> idxs_solve,
	cv::Mat& K, cv::Mat& R, cv::Mat& t, 
	double proj_estimation_threshold,
	double proj_estimation_threshold_weak,
	std::vector<int> &inliers,
	std::vector<int> &inliers_weak,
	std::vector<int> &outliers)
{
	/* First, find the projection matrix */
	printf("[FindAndVerifyCamera] find projection matrix\n");
	double P[12];
	int r = -1;

	v3_t *points_solve = new v3_t[num_points];
	v2_t *projs_solve = new v2_t[num_points];

	for (int i = 0 ;i < num_points; i++)
	{
		Vx(points_solve[i]) = points3d[i].x;
		Vy(points_solve[i]) = points3d[i].y;
		Vz(points_solve[i]) = points3d[i].z;


		Vx(projs_solve[i]) = points2d[i].x;
		Vy(projs_solve[i]) = points2d[i].y;
	}

	if (num_points >= 9) {
		r = find_projection_3x4_ransac(num_points, 
			points_solve, projs_solve, 
			P, /* 2048 */ 4096 /* 100000 */, 
			proj_estimation_threshold);
	}

	if (r == -1) {
		printf("[FindAndVerifyCamera] Couldn't find projection matrix\n");
		return false;
	}

	/* If number of inliers is too low, fail */
	if (r <= MIN_INLIERS_EST_PROJECTION) {
		printf("[FindAndVerifyCamera] Too few inliers to use "
			"projection matrix\n");
		return false;
	}

	double KRinit[9], Kinit[9], Rinit[9], tinit[3];
	memcpy(KRinit + 0, P + 0, 3 * sizeof(double));
	memcpy(KRinit + 3, P + 4, 3 * sizeof(double));
	memcpy(KRinit + 6, P + 8, 3 * sizeof(double));

	printf("[FindAndVerifyCamera] Compute an RQ factorization of the projection matrix. \n");

	dgerqf_driver(3, 3, KRinit, Kinit, Rinit);	    

	/* We want our intrinsics to have a certain form */
	FixIntrinsics(P, Kinit, Rinit, tinit);
	matrix_scale(3, 3, Kinit, 1.0 / Kinit[8], Kinit);

	//printf("[FindAndVerifyCamera] Estimated intrinsics:\n");
	//matrix_print(3, 3, Kinit);
	//printf("[FindAndVerifyCamera] Estimated extrinsics:\n");
	//matrix_print(3, 3, Rinit);
	//matrix_print(1, 3, tinit);
	//fflush(stdout);
	/* Check cheirality constraint */
	printf("[FindAndVerifyCamera] Checking consistency...\n");

	double Rigid[12] = 
	{ Rinit[0], Rinit[1], Rinit[2], tinit[0],
	Rinit[3], Rinit[4], Rinit[5], tinit[1],
	Rinit[6], Rinit[7], Rinit[8], tinit[2] };

	int num_behind = 0;
	for (int j = 0; j < num_points; j++) {
		double p[4] = { Vx(points_solve[j]), 
			Vy(points_solve[j]),
			Vz(points_solve[j]), 1.0 };
		double q[3], q2[3];

		matrix_product(3, 4, 4, 1, Rigid, p, q);
		matrix_product331(Kinit, q, q2);

		double pimg[2] = { -q2[0] / q2[2], -q2[1] / q2[2] };
		double diff = 
			(pimg[0] - Vx(projs_solve[j])) * 
			(pimg[0] - Vx(projs_solve[j])) + 
			(pimg[1] - Vy(projs_solve[j])) * 
			(pimg[1] - Vy(projs_solve[j]));

		diff = sqrt(diff);

		if (diff < proj_estimation_threshold)
			inliers.push_back(j);

		if (diff < proj_estimation_threshold_weak) {
			inliers_weak.push_back(j);
		} else {
			printf("[FindAndVerifyCamera] Removing point [%d] "
				"(reproj. error = %0.3f)\n", idxs_solve[j], diff);
			outliers.push_back(j);
		}

		if (q[2] > 0.0)
			num_behind++;  /* Cheirality constraint violated */
	}

	if (num_behind >= 0.9 * num_points) {
		printf("[FindAndVerifyCamera] Error: camera is pointing "
			"away from scene\n");
		return false;
	}

	cv::Mat Ks = cv::Mat(3,3,CV_64F,Kinit);
	cv::Mat Rmat = cv::Mat(3,3,CV_64F,Rinit);
	cv::Mat Tmat = cv::Mat(1,3,CV_64F,tinit);
	
	Ks.copyTo(K);
	Rmat.copyTo(R);
	Tmat.copyTo(t);

// 	std::cout<<std::endl;
// 	std::cout<<"Solved result: "<<std::endl;
// 	std::cout<<"K: "<<Ks<<std::endl;
// 	std::cout<<"R: "<<Rmat<<std::endl;
// 	std::cout<<"T: "<<Tmat<<std::endl;

	delete []points_solve;
	delete []projs_solve;
	return true;
}

SOLVEPNPLIB_API bool RefineCameraAndPoints(int num_points, 
	std::vector<cv::Point3f> points3d,	std::vector<cv::Point2f> points2d,
	std::vector<int> idxs_solve,camera_params_t& camera,
	std::vector<ImageKeyVector> &pt_views,std::vector<int> &inliers)
{
	return true;
}

//////////////////////////////////////////////////////////////////////////

SOLVEPNPLIB_API void InitializeCameraParams(camera_params_t &camera)
{
	matrix_ident(3, camera.R);
	camera.t[0] = camera.t[1] = camera.t[2] = 0.0;
	camera.f = 0.0;
	camera.k[0] = camera.k[1] = 0.0;

	camera.k_inv[0] = camera.k_inv[2] = camera.k_inv[3] = 0.0;
	camera.k_inv[4] = camera.k_inv[5] = 0.0;
	camera.k_inv[1] = 1.0;

	camera.f_scale = 1.0;
	camera.k_scale = 1.0;

	camera.known_intrinsics = 0;

	for (int i = 0; i < NUM_CAMERA_PARAMS; i++) {
		camera.constrained[i] = 0;
		camera.constraints[i] = 0.0;
		camera.weights[i] = 0.0;
	}

	 camera.fisheye = 0.0;
}

SOLVEPNPLIB_API void GetIntrinsics(const camera_params_t &camera, double *K) {
	if (!camera.known_intrinsics) {
		K[0] = camera.f;  K[1] = 0.0;       K[2] = 0.0;
		K[3] = 0.0;       K[4] = camera.f;  K[5] = 0.0;
		K[6] = 0.0;       K[7] = 0.0;       K[8] = 1.0;    
	} else {
		memcpy(K, camera.K_known, 9 * sizeof(double));
	}
}

SOLVEPNPLIB_API double GetCameraDistance(camera_params_t *c1, camera_params_t *c2)
{
	double center1[3]; 
	double Rinv1[9];
	matrix_invert(3, c1->R, Rinv1);

	memcpy(center1, c1->t, 3 * sizeof(double));

	double center2[3];
	double Rinv2[9];
	matrix_invert(3, c2->R, Rinv2);

	memcpy(center2, c2->t, 3 * sizeof(double));

	double dx = center1[0] - center2[0];
	double dy = center1[1] - center2[1];
	double dz = center1[2] - center2[2];

	return sqrt(dx * dx + dy * dy + dz * dz);
}

v2_t UndistortNormalizedPoint(v2_t p, camera_params_t c) 
{
	double r = sqrt(Vx(p) * Vx(p) + Vy(p) * Vy(p));
	if (r == 0.0)
		return p;

	double t = 1.0;
	double a = 0.0;

	for (int i = 0; i < POLY_INVERSE_DEGREE; i++) {
		a += t * c.k_inv[i];
		t = t * r;
	}

	double factor = a / r;

	return v2_scale(factor, p);
}

SOLVEPNPLIB_API cv::Point2f UndistortNormPoint(cv::Point2f p, camera_params_t c)
{
	v2_t q ;
	Vx(q) = p.x, Vy(q) = p.y;

	v2_t norm_q =  UndistortNormalizedPoint(q,c);

	return cv::Point2f(Vx(norm_q),Vy(norm_q));
}

//////////////////////////////////////////////////////////////////////////

/* Compute the angle between two rays */
double ComputeRayAngle(v2_t p, v2_t q, 
	const camera_params_t &cam1, 
	const camera_params_t &cam2)
{
	double K1[9], K2[9];
	GetIntrinsics(cam1, K1);
	GetIntrinsics(cam2, K2);

	double K1_inv[9], K2_inv[9];
	matrix_invert(3, K1, K1_inv);
	matrix_invert(3, K2, K2_inv);

	double p3[3] = { Vx(p), Vy(p), 1.0 };
	double q3[3] = { Vx(q), Vy(q), 1.0 };

	double p3_norm[3], q3_norm[3];
	matrix_product331(K1_inv, p3, p3_norm);
	matrix_product331(K2_inv, q3, q3_norm);

	v2_t p_norm = v2_new(p3_norm[0] / p3_norm[2], p3_norm[1] / p3_norm[2]);
	v2_t q_norm = v2_new(q3_norm[0] / q3_norm[2], q3_norm[1] / q3_norm[2]);

	double R1_inv[9], R2_inv[9];
	matrix_transpose(3, 3, (double *) cam1.R, R1_inv);
	matrix_transpose(3, 3, (double *) cam2.R, R2_inv);

	double p_w[3], q_w[3];

	double pv[3] = { Vx(p_norm), Vy(p_norm), -1.0 };
	double qv[3] = { Vx(q_norm), Vy(q_norm), -1.0 };

	double Rpv[3], Rqv[3];

	matrix_product331(R1_inv, pv, Rpv);
	matrix_product331(R2_inv, qv, Rqv);

	matrix_sum(3, 1, 3, 1, Rpv, (double *) cam1.t, p_w);
	matrix_sum(3, 1, 3, 1, Rqv, (double *) cam2.t, q_w);

	/* Subtract out the camera center */
	double p_vec[3], q_vec[3];
	matrix_diff(3, 1, 3, 1, p_w, (double *) cam1.t, p_vec);
	matrix_diff(3, 1, 3, 1, q_w, (double *) cam2.t, q_vec);

	/* Compute the angle between the rays */
	double dot;
	matrix_product(1, 3, 3, 1, p_vec, q_vec, &dot);

	double mag = matrix_norm(3, 1, p_vec) * matrix_norm(3, 1, q_vec);

	return acos(CLAMP(dot / mag, -1.0 + 1.0e-8, 1.0 - 1.0e-8));
}

/* Check cheirality for a camera and a point */
bool CheckCheirality(v3_t p, const camera_params_t &camera) 
{
	double pt[3] = { Vx(p), Vy(p), Vz(p) };
	double cam[3];

	pt[0] -= camera.t[0];
	pt[1] -= camera.t[1];
	pt[2] -= camera.t[2];
	matrix_product(3, 3, 3, 1, (double *) camera.R, pt, cam); // RX + T

	if (cam[2] > 0.0)
		return false;
	else
		return true;
}

/* Triangulate two points */
v3_t Triangulate(v2_t p, v2_t q, 
	camera_params_t c1, camera_params_t c2, 
	double &proj_error, bool &in_front, double &angle
/*	,bool explicit_camera_centers*/)
{
	double K1[9], K2[9];
	double K1inv[9], K2inv[9];
	v3_t pt;

	GetIntrinsics(c1, K1);
	GetIntrinsics(c2, K2);

	if(K1[0] < 1e-5 || K2[0] > 1e5)
		return pt;

	matrix_invert(3, K1, K1inv);
	matrix_invert(3, K2, K2inv);

	/* Set up the 3D point */
	double proj1[3] = { Vx(p), Vy(p), 1.0 };
	double proj2[3] = { Vx(q), Vy(q), 1.0 };

	double proj1_norm[3], proj2_norm[3];

	matrix_product(3, 3, 3, 1, K1inv, proj1, proj1_norm);
	matrix_product(3, 3, 3, 1, K2inv, proj2, proj2_norm);

//	printf("%f,%f\n",proj1_norm[2],proj2_norm[2]);

	v2_t p_norm = v2_new(-proj1_norm[0],	-proj1_norm[1]);
	v2_t q_norm = v2_new(-proj2_norm[0],	-proj2_norm[1]);

	/* Compute the angle between the rays */
	angle = ComputeRayAngle(p, q, c1, c2);

	/* Undo radial distortion */
	//p_norm = UndistortNormalizedPoint(p_norm, c1);
	//q_norm = UndistortNormalizedPoint(q_norm, c2);

	/* Triangulate the point */
	
	{
		double t1[3];
		double t2[3];
		/* Put the translation in standard form */
		matrix_product(3, 3, 3, 1, c1.R, c1.t, t1);
		matrix_scale(3, 1, t1, -1.0, t1);
		matrix_product(3, 3, 3, 1, c2.R, c2.t, t2);
		matrix_scale(3, 1, t2, -1.0, t2);

		pt = triangulate(p_norm, q_norm, c1.R, t1, c2.R, t2, &proj_error);
	}

	

	proj_error = (c1.f + c2.f) * 0.5 * sqrt(proj_error * 0.5);

	/* Check cheirality */
	bool cc1 = CheckCheirality(pt, c1);
	bool cc2 = CheckCheirality(pt, c2);

	in_front = (cc1 && cc2);

	return pt;
}


double m_projection_estimation_threshold = 4.0;

SOLVEPNPLIB_API int Triangulate(std::vector<cv::Point2f> l_pt,std::vector<cv::Point2f> r_pt,int num_points,
	int matches[][2],camera_params_t l_camera, camera_params_t r_camera,std::vector<cv::Point3f>& points ,std::vector<int>& indexs)
{
	points.clear();
	indexs.clear();

	v2_t p,q;
	
	double reproj_err = 0;

//	std::ofstream fp("Triangulate.txt");
	int num_inliers = 0;
	for (int i = 0 ; i < num_points; i++)
	{
		int m_idx1 = matches[i][0];
		int m_idx2 = matches[i][1];

		cv::Point2f pt1 = l_pt[m_idx1];
		cv::Point2f pt2 = r_pt[m_idx2];

		Vx(p) = pt1.x, Vy(p) = pt1.y;
		Vx(q) = pt2.x, Vy(q) = pt2.y;

		double prj_err, angle;
		bool in_front;

		v3_t pt3d = Triangulate(p,q,l_camera,r_camera,prj_err,in_front,angle);

		//printf(" tri.error[%d] = %0.3f\n", i, prj_err);
		
		if (prj_err > /*4.0*/ m_projection_estimation_threshold) {
			//printf(" skipping point\n");
			continue;
		}

		if(in_front)
		{
			reproj_err += prj_err;
			num_inliers++;
			//fp << Vx(pt3d)<<" "<<Vy(pt3d)<<" "<<Vz(pt3d)<<" "<<prj_err<<" "<<in_front<<" "<<angle<<std::endl;
			points.push_back(cv::Point3f(Vx(pt3d),Vy(pt3d),Vz(pt3d)));
			indexs.push_back(i);
		}	
	}

//	fp.close();
	if(num_inliers > 0 )
		reproj_err /= num_inliers;

	printf("[Triangulate] re_projection error: %f\n",reproj_err);

	return num_inliers;
}

SOLVEPNPLIB_API bool Triangulate(cv::Point2f l_pt, cv::Point2f r_pt,
	camera_params_t l_camera, camera_params_t r_camera,cv::Point3f & point)
{
	v2_t p,q;

	Vx(p) = l_pt.x, Vy(p) = l_pt.y;
	Vx(q) = r_pt.x, Vy(q) = r_pt.y;

	double prj_err, angle;
	bool in_front;

	v3_t pt3d = Triangulate(p,q,l_camera,r_camera,prj_err,in_front,angle);

	printf("[Triangulate] tri.error = %0.3f\n", prj_err);

	point.x = Vx(pt3d);
	point.y = Vy(pt3d);
	point.z = Vz(pt3d);

	return in_front;
}

//////////////////////////////////////////////////////////////////////////
// Found Pose
SOLVEPNPLIB_API bool EstimatePose(std::vector<cv::Point2f> keys1, std::vector<cv::Point2f> keys2,
	int match_num,int matches[][2],camera_params_t &c1, camera_params_t &c2)
{
	std::vector<KeypointMatch> keyMatches; 
	for (int i = 0; i < match_num; i++)
	{
		int m_idx1 = matches[i][0];
		int m_idx2 = matches[i][1];

		KeypointMatch m(m_idx1,m_idx2);
		keyMatches.push_back(m);
	}

	std::vector<Keypoint> pts1,pts2;

	for(int i = 0 ; i < keys1.size(); i++ )
	{
		Keypoint pt(keys1[i].x,keys1[i].y);
		pts1.push_back(pt);
	}

	for (int i = 0 ; i < keys2.size(); i++ )
	{
		Keypoint pt(keys2[i].x,keys2[i].y);
		pts2.push_back(pt);
	}

	// Test input data
//	printf("idx: %d, match : %d,%d\n",50,keyMatches[50].m_idx1,keyMatches[50].m_idx2);

	double K1[9], K2[9];
	GetIntrinsics(c1, K1);
	GetIntrinsics(c2, K2);

	double R0[9], t0[3];
	int num_inliers = 0;

	num_inliers = 
		EstimatePose5Point(pts1, 
		pts2, 
		keyMatches,
		512, /* m_fmatrix_rounds, 8 * m_fmatrix_rounds */
		0.25 * 9,
		K1, K2, R0, t0);

	if (num_inliers == 0)
		return false;

	// confirm E = [t]x * R, F = K^-T*E*K^-1
/*

	float detR = matrix_determinant3(R0);
	float detT = matrix_norm(3,3,t0);
	double tx[9] = { 0,-t0[2],t0[1],
					t0[2],0,-t0[0],
					-t0[1],t0[0],0};

	double E[9],Rt[9];
	matrix_product(3,3,3,3,tx,R0,E);
	double detE = matrix_determinant3(E);
	printf("detR = %f,detT = %f, E[8] = %f,detE = %lf\n",detR,detT,E[8],detE);

	printf("t0: %f,%f,%f\n",t0[0],t0[1],t0[2]);
	printf("R0: \n");
	matrix_print(3,3,R0);
	matrix_scale(3, 3, E, 1.0/E[8], E);
	printf("Test E: \n");
	matrix_print(3,3,E);
*/
// 	 double K1_inv[9], K2_inv[9];
// 	 matrix_invert(3, K1, K1_inv);
// 	 matrix_invert(3, K2, K2_inv);
// 	 double F[9];
// 	 double tmp[9];
// 	 matrix_product33(K2_inv,E,tmp);
// 	 matrix_product33(tmp,K1_inv,F);
// 	 printf("Test F: \n");
// 	 matrix_print(3,3,F);
		
	 bool initialized = false;
	 if (!initialized) {
		 memcpy(c2.R, R0, sizeof(double) * 9);
		 matrix_transpose_product(3, 3, 3, 1, R0, t0, c2.t);
		 matrix_scale(3, 1, c2.t, -1.0, c2.t);  //-R^T*t
	 }
	
	
	return true;
}