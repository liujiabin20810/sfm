#ifndef  BUNDLER_COMMON_HPP__
#define BUNDLER_COMMON_HPP__
#pragma once

#include <opencv2/core/core.hpp>

namespace bundler
{
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
			float m_color[3]; /* Color of the point */
			double m_conf;    /* Confidence in this point */

			ImageKeyVector m_views;  /* View / keys corresponding to this point */
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

		std::vector<cv::Mat> m_images;
		std::vector<std::string>  m_imageNameList;

		cv::Mat m_K, m_Kinv, m_distortion_coeff;
		cv::Mat m_distcoeff_32f;
		cv::Mat m_K_32f;
	};
}

#endif