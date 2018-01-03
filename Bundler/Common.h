#ifndef  BUNDLER_COMMON_HPP__
#define BUNDLER_COMMON_HPP__
#pragma once

#include <opencv2/core/core.hpp>

namespace bundler
{
		#define NUM_CAMERA_PARAMS 9
		#define POLY_INVERSE_DEGREE 6

		typedef struct {
			double R[9];     /* Rotation */
			double t[3];     /* Translation */
			double f;        /* Focal length */
			double k[2];     /* Undistortion parameters */
			double k_inv[POLY_INVERSE_DEGREE]; /* Inverse undistortion parameters */
			char constrained[NUM_CAMERA_PARAMS];
			double constraints[NUM_CAMERA_PARAMS];  /* Constraints (if used) */
			double weights[NUM_CAMERA_PARAMS];      /* Weights on the constraints */
			double K_known[9];  /* Intrinsics (if known) */
			double k_known[5];  /* Distortion params (if known) */

			char fisheye;            /* Is this a fisheye image? */
			char known_intrinsics;   /* Are the intrinsics known? */
			double f_cx, f_cy;       /* Fisheye center */
			double f_rad, f_angle;   /* Other fisheye parameters */
			double f_focal;          /* Fisheye focal length */

			double f_scale, k_scale; /* Scale on focal length, distortion params */
		} camera_params_t;

		struct CloudPoint
		{
			cv::Point3d pt3d;
			cv::Scalar pt3d_color;
			int ith_camera;
			int jth_camera;
			std::vector<unsigned int> pt2d_index_of_img;
			double reprojection_error;
		};

		typedef std::pair<int,int> ImageKey;
		typedef std::vector<ImageKey> ImageKeyVector;

		typedef struct
		{
			int image;
			int key;
			double x;
			double y;
		} view_t;

		/* Data for 3D points */
		class PointData
		{
		 public:
			PointData() { m_fixed = false; }

			/* Write the point data in XML */
			void WriteXML(FILE *f);
			void WriteGeoXML(FILE *f);

			/* Write coordinates*/
			void WriteCoordinates(FILE *f);

			/* Create a planar patch for this point */
			//void CreatePlanarPatch(double size, PlaneData &plane);

			double m_pos[3];  /* 3D position of the point */
			double m_norm[3]; /* Estimated normal for this point */
			int m_color[3]; /* Color of the point */
			double m_conf;    /* Confidence in this point */

			std::vector<view_t> m_views;  /* View / keys corresponding to this point */
			bool m_fixed;      /* Should this point be fixed during bundle
					* adjustment? */

			float *m_desc;     /* Descriptor for this point */
			int m_num_vis;     /*number of images that see this point*/
			int m_ref_image;   /* Reference image */
		};

	class Utils
	{
	public:

		Utils();
		~Utils();

		bool hasEnding(std::string const &fullString, std::string const &ending);
		bool hasEndingLower(std::string const &fullString_, std::string const &_ending);
		int open_imgs_dir(std::string dir_name_);

		int findImage(char strPath[]);
		int loadCalibMatrix(char strPath[]);

		void WriteBundleFile(const char *bundle_file, 
			const std::vector<camera_params_t> &cameras,
			std::vector<int> added_order,
			const std::vector<PointData> &points);

		void ReadBundleFile(const char *bundle_file, 
			std::vector<camera_params_t> &cameras,
			std::vector<PointData> &points, double &bundle_version);

		void WritePMVS(const char *output_path, 
			std::vector<std::string> images, 
			std::vector<camera_params_t> &cameras,
			std::vector<int> added_order,
			const std::vector<PointData> &points);

		std::vector<cv::Mat> m_images;
		std::vector<std::string>  m_imageNameList;

		int m_image_width, m_image_height;

		cv::Mat m_K, m_Kinv, m_distortion_coeff;
		cv::Mat m_distcoeff_32f;
		cv::Mat m_K_32f;
	};
}

#endif