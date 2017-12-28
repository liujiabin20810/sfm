
/*MatchTracks.cpp*/
/* Code for processing matches and tracks */
#include "MatchTracks.h"

#include <opencv2/calib3d/calib3d.hpp>

#include <queue>
#include <iostream>
#include <stdio.h>

#define  ransac_stop_support 50
#define  RANSAC_DIST_THRESHOLD 5

#define  USING_OPENCV

namespace bundler
{
	/* Get match information */
	MatchIndex GetMatchIndex(int i1, int i2) 
	{
		// MatchIndex num_images = GetNumImages();
		// return i1 * num_images + i2;
		return MatchIndex((unsigned long) i1, (unsigned long) i2);
	}	

	MatchIndex GetMatchIndexUnordered(int i1, int i2) 
	{
		// MatchIndex num_images = GetNumImages();
		// return i1 * num_images + i2;
		if (i1 < i2)
			return MatchIndex((unsigned long) i1, (unsigned long) i2);
		else
			return MatchIndex((unsigned long) i2, (unsigned long) i1);
	}

	/* Return the intersection of two int vectors */
	std::vector<int> MatchTracks::GetVectorIntersection(const std::vector<int> &v1,
						   const std::vector<int> &v2)
	{
		stdext::hash_set<int> seen;
		int v1_size = (int) v1.size();
		int v2_size = (int) v2.size();

		std::vector<int> intersection;

		for (int i = 0; i < v1_size; i++)
			seen.insert(v1[i]);

		for (int i = 0; i < v2_size; i++) 
		{
			if (seen.find(v2[i]) != seen.end())
				intersection.push_back(v2[i]);
		}

		seen.clear();

		return intersection;
	}

	void TransformInfo::ReadFromFile(FILE *f)
	{
		/* Homography */
		fscanf(f, "%lf %lf %lf "
			  "%lf %lf %lf "
			  "%lf %lf %lf",
		   m_H + 0, m_H + 1, m_H + 2,
		   m_H + 3, m_H + 4, m_H + 5,
		   m_H + 6, m_H + 7, m_H + 8);

		/* F-Matrix */
		fscanf(f, "%lf %lf %lf "
			  "%lf %lf %lf "
			  "%lf %lf %lf",
		   m_fmatrix + 0, m_fmatrix + 1, m_fmatrix + 2,
		   m_fmatrix + 3, m_fmatrix + 4, m_fmatrix + 5,
		   m_fmatrix + 6, m_fmatrix + 7, m_fmatrix + 8);


		/* Inlier info */
		fscanf(f, "%lf\n", &m_inlier_ratio);
		fscanf(f, "%d\n", &m_num_inliers); 
	}

	void TransformInfo::WriteToFile(FILE *f)
	{
		/* Homography */
		fprintf(f, "%0.6le %0.6le %0.6le "
			   "%0.6le %0.6le %0.6le "
			   "%0.6le %0.6le %0.6le\n",
			m_H[0], m_H[1], m_H[2],
			m_H[3], m_H[4], m_H[5],
			m_H[6], m_H[7], m_H[8]);
    
		/* F-Matrix */
		fprintf(f, "%0.6le %0.6le %0.6le "
			   "%0.6le %0.6le %0.6le "
			   "%0.6le %0.6le %0.6le\n",
			m_fmatrix[0], m_fmatrix[1], m_fmatrix[2],
			m_fmatrix[3], m_fmatrix[4], m_fmatrix[5],
			m_fmatrix[6], m_fmatrix[7], m_fmatrix[8]);

		/* Inlier info */
		fprintf(f, "%0.16le\n", m_inlier_ratio);
		fprintf(f, "%d\n", m_num_inliers);
	}

	void TrackData::Read(FILE *f)
	{
		int size;
		fscanf(f, "%d", &size);
    
		for (int i = 0; i < size; i++) {
			
			ImageKey ik;
			fscanf(f, "%d %d", &(ik.first), &(ik.second));
			m_views.push_back(ik);
		}
	}

	void TrackData::Write(FILE *f)
	{
		int size = (int) m_views.size();
		fprintf(f, "%d", size);
    
		for (int i = 0; i < size; i++) {
		
			fprintf(f, " %d %d", m_views[i].first, m_views[i].second);
		}

		fprintf(f, "\n");
	}

	MatchTracks::MatchTracks(void)
	{
		m_min_num_feat_matches = 50;
	}

	void MatchTracks::RemoveAllMatches()
	{
		m_matches_table.RemoveAll();
	}

	bool MatchTracks::ImagesMatch(int i1, int i2)
	{
		return m_matches_table.Contains(GetMatchIndex(i1,i2));
	}

	void MatchTracks::SetMatch(int i1, int i2)
	{
		m_matches_table.SetMatch(GetMatchIndex(i1, i2));
	}

	bool CompareFirst(const KeypointMatch &k1, const KeypointMatch &k2) 
	{
		return (k1.m_idx1 < k2.m_idx1);
	}

	void MatchTracks::InitMatchTable(std::map<std::pair<int, int>, std::vector<cv::DMatch> > _matches_matrix,
		std::vector<std::vector<cv::KeyPoint>> _images_points ,
		std::map<std::pair<int, int>, cv::Mat > _fmatrix, 
		std::vector<cv::Mat> _images,
		int num_images)
	{
		m_matches_table = MatchTable(num_images);
		RemoveAllMatches();


		// m_image_data init
		m_image_data.resize(num_images);

		//m_images_points = _images_points;
		for (int i = 0; i < num_images; i++)
		{
			std::vector<cv::KeyPoint> points = _images_points[i];
			int point_num = points.size();
			m_image_data[i].m_keys.resize(point_num);

			for (int j = 0; j < point_num; j++)
			{
				cv::KeyPoint point = points[j];

				float x = point.pt.x, y = point.pt.y;
				cv::Vec3b _color =  _images[i].ptr<cv::Vec3b>((int)(y+0.5))[(int)(x+0.5)];

				m_image_data[i].m_keys[j] = Keypoint(x, y);
				m_image_data[i].m_keys[j].m_b = _color[0];
				m_image_data[i].m_keys[j].m_g = _color[1];
				m_image_data[i].m_keys[j].m_r = _color[2];

				m_image_data[i].m_keys[j].m_extra = -1;
			}

		}

		m_num_images = num_images;
		m_matches_matrix = _matches_matrix;	
		
		// F matrixs not used
		m_fmatrix = _fmatrix;

		m_images =  _images;

		m_matches_computed = true;
		m_matches_loaded = false;

		m_min_track_views = 2;
		m_max_track_views = 10000;

	}

	void MatchTracks::PruneDoubleMatches()
	{
		for (unsigned int i = 0; i < m_num_images; i++) 
		{

			MatchAdjList::iterator iter;
			std::vector<unsigned int> remove;
			for (iter = m_matches_table.Begin(i); iter != m_matches_table.End(i); iter++)
			{
				HashSetInt seen;

				int num_pruned = 0;
				// MatchIndex idx = *iter; // GetMatchIndex(i, j);
				std::vector<KeypointMatch> &list = iter->m_match_list;

				/* Unmark keys */
				// int num_matches = (int) m_match_lists[idx].size();
				int num_matches = (int) list.size();

				for (int k = 0; k < num_matches; k++)
				{
					int idx2 = list[k].m_idx2;
		
					// if (GetKey(j,idx2).m_extra != -1) {
					if (seen.find(idx2) != seen.end()) 
					{
						/* This is a repeat */
						// printf("[%d] Pruning repeat %d\n", i, idx2);
						list.erase(list.begin() + k);
						num_matches--;
						k--;
                    
						num_pruned++;
					 } 
					else 
					{
						/* Mark this key as matched */
						// GetKey(j,idx2).m_extra = k;
						 seen.insert(idx2);
					 }
				 }

				// unsigned int i = iter->first;
				// unsigned int j = iter->second;
				unsigned int j = iter->m_index; // first;

				if(num_pruned > 0 )
					printf("[PruneDoubleMatches] Pruned[%d,%d] = %d / %d\n",
					   i, j, num_pruned, num_matches + num_pruned);

				if (num_matches < m_min_num_feat_matches) {
					/* Get rid of... */
					remove.push_back(iter->m_index); // first);
				}
			} // for iter
////////////////////////////////////////////////////////////
			for (unsigned int j = 0; j < remove.size(); j++) 
			{
				int idx2 = remove[j];
				m_matches_table.RemoveMatch(GetMatchIndex(i, idx2));
				printf("[PruneDoubleMatches] Removing[%d,%d]\n", i, idx2);
			}
////////////////////////////////////////////////////////////
		}// for i
	
	}

	void MatchTracks::CreateMatchTable()
	{
		if (m_matches_loaded)
			return;  /* we already loaded the matches */

		std::cout<<"[LoadMatchTable] Loading matches"<<std::endl;
		RemoveAllMatches();

		std::map<std::pair<int, int>, std::vector<cv::DMatch> >::iterator matrix_iter = m_matches_matrix.begin();

		for(; matrix_iter != m_matches_matrix.end(); ++matrix_iter)
		{
			int i1, i2;
			i1 = (matrix_iter->first).first;
			i2 = (matrix_iter->first).second;
			
			std::vector<cv::DMatch> dmatch = matrix_iter->second;
			std::vector<cv::DMatch>::iterator dmatch_iter = dmatch.begin();
			std::vector<KeypointMatch> matches;
			for(;dmatch_iter != dmatch.end(); ++dmatch_iter )
			{
				KeypointMatch m;

				m.m_idx1 = dmatch_iter->queryIdx;
				m.m_idx2 = dmatch_iter->trainIdx;
				
				matches.push_back(m);
			}

			MatchIndex idx = GetMatchIndex(i1, i2);
			m_matches_table.SetMatch(idx);
			m_matches_table.GetMatchList(idx) = matches;
		}

		PruneDoubleMatches();

		m_matches_loaded = true;
	}

	void MatchTracks::MakeMatchListsSymmetric()
	{
		unsigned int num_images = m_num_images;

		std::vector<MatchIndex> matches;

		for (unsigned int i = 0; i < num_images; i++) 
		{
			MatchAdjList::const_iterator iter;
			for (iter = m_matches_table.Begin(i); iter != m_matches_table.End(i); iter++) {
				// unsigned int i = iter->first;
				unsigned int j = iter->m_index; // iter->second;

				if (j <= i)
					continue;

				assert(m_matches_table.Contains(GetMatchIndex(i, j)));   

				// MatchIndex idx = *iter; 
				MatchIndex idx = GetMatchIndex(i, j);
				MatchIndex idx_rev = GetMatchIndex(j, i);

				const std::vector<KeypointMatch> &list = iter->m_match_list;
				unsigned int num_matches = list.size();

				m_matches_table.SetMatch(idx_rev);
				m_matches_table.ClearMatch(idx_rev);

				for (unsigned int k = 0; k < num_matches; k++) {
					KeypointMatch m1, m2;
		
					m1 = list[k];
                
					m2.m_idx1 = m1.m_idx2;
					m2.m_idx2 = m1.m_idx1;

					// m_match_lists[idx_rev].push_back(m2);
					m_matches_table.AddMatch(idx_rev, m2);
				}

				matches.push_back(idx);
			}
		}

		unsigned int num_matches = matches.size();

		for (unsigned int i = 0; i < num_matches; i++) {
			unsigned int img1 = matches[i].first;
			unsigned int img2 = matches[i].second;
			SetMatch(img2, img1);
		}

		//////////////////////////////////////////////////////////////////////////

// 		for (unsigned int i = 0; i < num_matches; i++) {
// 			unsigned int img1 = matches[i].first;
// 			unsigned int img2 = matches[i].second;
// 
// 			MatchIndex idx = GetMatchIndex(img1, img2);
// 			MatchIndex idx_rev = GetMatchIndex(img2, img1);
// 			int sz1 = m_matches_table.GetNumMatches(idx);
// 			int sz2 = m_matches_table.GetNumMatches(idx_rev); 
// 			
// 			const std::vector<KeypointMatch> &list1 = m_matches_table.GetMatchList(idx);
// 			const std::vector<KeypointMatch> &list2 = m_matches_table.GetMatchList(idx_rev);
// 
// 			if(img1 == 2 && img2 == 3)
// 			{
// 				for(int k = 0; k< sz1; k++)
// 				{
// 					if(k%50 == 0)
// 						printf("[%d,%d] ",list1[k].m_idx1,list1[k].m_idx2);
// 				}
// 				printf("\n");
// 				for(int k = 0; k< sz2; k++)
// 				{
// 					if(k%50 == 0)
// 						printf("[%d,%d] ",list2[k].m_idx1,list2[k].m_idx2);
// 				}
// 			}
// 
// 			printf("[MatchListsSymmetric] Match(%d,%d): %d;		Match(%d,%d): %d. \n",img1,img2,sz1,img2,img1,sz2);
// 			//std::cout<< "["<<img1<<","<<img2<<"]: "<<sz1<<" ["<<img2<<","<<img1<<"]: "<<sz2<<std::endl;
// 		}

		matches.clear();

	}

	/* Compute a set of tracks that explain the matches */
	void MatchTracks::ComputeTracks(int new_image_start)
	{
		int num_images = m_num_images;

		/* Clear all marks for new images */
		//std::vector<std::vector<bool> > m_key_flags;
		for (unsigned int i = 0; i < num_images; i++)
		{
			int num_nbrs = (int) m_matches_table.GetNumNeighbors(i);

			if (num_nbrs == 0)
				continue;

			//std::vector<bool> key_flags;
			int num_features = m_image_data[i].m_keys.size();
			m_image_data[i].m_key_flags.resize(num_features);

			//m_key_flags.push_back(key_flags);
		 }

		/* Sort all match lists */
		for (unsigned int i = 0; i < num_images; i++)
		{
			 MatchAdjList::iterator iter;
			for (iter = m_matches_table.Begin(i); iter != m_matches_table.End(i); iter++) {
				// MatchIndex idx = *iter;
				std::vector<KeypointMatch> &list = iter->m_match_list; // iter->second; // m_match_lists[idx];
				sort(list.begin(), list.end(), CompareFirst);
				//std::cout<< list[0].m_idx1 <<" "<<list[1].m_idx1<<std::endl;
#if 0				
				for (int j = 0; j < list.size(); j++)
				{
					std::cout<< list[j].m_idx1<<" ";
				}
				std::cout<<std::endl;
#endif
			}
		}

		int pt_idx = 0;
		std::vector<TrackData> tracks;
		bool *img_marked = new bool[num_images];
		memset(img_marked, 0, num_images * sizeof(bool));

		std::vector<int> touched;
		touched.reserve(num_images);

		for (unsigned int i = 0; i < num_images; i++)
		{
			/* If this image has no neighbors, skip it */
			int num_nbrs = (int) m_matches_table.GetNumNeighbors(i);

			if (num_nbrs == 0)
				continue;

			int num_features = m_image_data[i].m_keys.size();
			int pt_unused = 0;
			int pt_used = 0;
			for (int j = 0; j < num_features; j++)
			{
				ImageKeyVector features;
				std::queue<ImageKey> features_queue;

				/* Check if this feature was visited */
				if (m_image_data[i].m_key_flags[j])
				{
					pt_used++;
					continue; // already visited this feature
				}

				int num_touched = touched.size();
				//std::cout<<"touched size: "<<num_touched<<std::endl;

				// touched[k] mark which image was markd as true, reset to false
				// It is only valid to same feature point and 
				// useful to find match_feature point in all adjacent images quickly
				for (int k = 0; k < num_touched; k++)
					img_marked[touched[k]] = false;
				touched.clear();

				m_image_data[i].m_key_flags[j] = true;

				// i,j : 表示第i幅图中第j个KeyPoint
				features.push_back(ImageKey(i, j));
				features_queue.push(ImageKey(i, j));

				img_marked[i] = true;
				touched.push_back(i);

				int num_rounds = 0;
				while(!features_queue.empty())
				{
					num_rounds++;

					ImageKey feature = features_queue.front();
					features_queue.pop();
		
					int img1 = feature.first;
					int f1 = feature.second;
					KeypointMatch dummy;
					dummy.m_idx1 = f1;

					int start_idx;
					/* Limit new images to point only to other new images */
					if (img1 >= new_image_start) {
						start_idx = new_image_start;
					} else {
						start_idx = 0;
					}

					MatchAdjList &nbrs = m_matches_table.GetNeighbors(img1);
					MatchAdjList::iterator iter;
					 /* Check all adjacent images */
					for(iter = nbrs.begin(); iter != nbrs.end(); ++iter)
					{
						// k image index
						unsigned int k = iter->m_index; // *iter; // nbrs[nbr];
						
						if (img_marked[k])
							 continue;

						MatchIndex base = GetMatchIndex(img1, k);

						std::vector<KeypointMatch> &list = 
                        m_matches_table.GetMatchList(base); // m_match_lists[base];

						/* Do a binary search for the feature */
						std::pair<std::vector<KeypointMatch>::iterator, 
								  std::vector<KeypointMatch>::iterator> p;

						p = equal_range(list.begin(), list.end(), 
										dummy, CompareFirst);

						if (p.first == p.second)
							continue;  /* not found */

						assert((p.first)->m_idx1 == f1);
						int idx2 = (p.first)->m_idx2;
			    
						/* Check if we visited this point already */
						// if (GetKey(k,idx2).m_extra >= 0)
						//     continue;
						assert(idx2 < m_key_flags[k].size());

						//marked match
						if (m_image_data[k].m_key_flags[idx2])
							continue;

						/* Mark and push the point */
						// GetKey(k,idx2).m_extra = pt_idx;
						m_image_data[k].m_key_flags[idx2] = true;
						features.push_back(ImageKey(k, idx2));
						features_queue.push(ImageKey(k, idx2));

						// once found the KeyPoint Match in the kth image
						// set mark to true ,no repeat in neighbors search
						img_marked[k] = true;
						touched.push_back(k);
					} // MatchAdjList::iterator iter
				}// while loop 
				 
				// track length >= 3, matches >= 2
				if (features.size() >= 2 )
				{
					//printf("Point with %d projections found\n",(int) features.size()); 
					//fflush(stdout);					
					tracks.push_back(TrackData(features));

					pt_idx++;
					pt_used++;
				}
				else
				{
					pt_unused++;
						
				}
			} // for j loop over features(KeyPoints)
			
		  //printf("[%d / %d] KeyPoints .\n",pt_used,pt_unused);
		} // for i loop over images


		 printf("[ComputeTracks] Found %d points\n", pt_idx);
		 fflush(stdout);

		 if (pt_idx != (int) tracks.size())
		 {
			printf("[ComputeTracks] Error: point count "
					"inconsistent!\n");
			fflush(stdout);
		}

		/* Clear match lists */
		printf("[ComputeTracks] Clearing match lists...\n");
		fflush(stdout);	
		RemoveAllMatches();

		/* Create the new consistent match lists */
		printf("[ComputeTracks] Creating consistent match lists...\n");
		fflush(stdout);
		int num_pts = pt_idx;

		for(int i = 0 ; i < num_pts; i++)
		{
			int num_features = (int) tracks[i].m_views.size();
			for(int j = 0 ; j < num_features; j++)
			{
				int im_index = tracks[i].m_views[j].first;
				int key_index = tracks[i].m_views[j].second;

				// m_image_data init
				m_image_data[im_index].m_visible_points.push_back(i);
				m_image_data[im_index].m_visible_keys.push_back(key_index);

			}

		}

		//for(int i = 0;i < m_num_images; i++)
		//{
		//	std::vector<int> tracks = m_image_data[i].m_visible_points;
		//	std::cout<<"image tracks "<<i<<" : ";
		//	for(int j = 0; j <tracks.size(); j++ )
		//	{
		//		std::cout<<tracks[j]<<" ";
		//	}

		//	std::cout<<std::endl;
		//}

		/* Save the tracks */
		m_track_data = tracks;

		printf("[ComputeTracks] Done!\n");
		fflush(stdout);
	}

	void MatchTracks::SetMatchesFromTracks()
	{
		RemoveAllMatches();

		int num_tracks = (int) m_track_data.size();

		int num_tracks_used = 0;
		for (int i = 0; i < num_tracks; i++)
		{
			TrackData &t = m_track_data[i];
	
			int num_views = (int) t.m_views.size();
	
			if (num_views < m_min_track_views) 
				continue; /* Not enough observations */

			if (num_views > m_max_track_views) 
				continue; /* Too many observations */

			for (int j = 0; j < num_views; j++) 
			{
				int v1 = t.m_views[j].first;
				int k1 = t.m_views[j].second;

				 assert(v1 >= 0 && v1 < num_images);
				for (int k = 0; k < num_views; k++)
				{
					if (j == k) continue;

					int v2 = t.m_views[k].first;
					 assert(v2 >= 0 && v2 < num_images);
					int k2 = t.m_views[k].second;
		
					MatchIndex idx = GetMatchIndex(v1, v2);
					m_matches_table.SetMatch(idx);
					m_matches_table.AddMatch(idx,KeypointMatch(k1, k2));
				}
			}

			num_tracks_used++;
		}

		printf("[BaseApp::SetMatchesFromTracks] Used %d tracks\n", 
			   num_tracks_used);
	}

	// 1. find same indexes between img1 and img2
	// 2. find the positions of the same indexes
	// 3. Update KeypointMatch vector
	void MatchTracks::SetMatchesFromTracks(int img1, int img2)
	{
		std::vector<int> &track1 = m_image_data[img1].m_visible_points;
		std::vector<int> &track2 = m_image_data[img2].m_visible_points;

		// 1.
		std::vector<int> isect = GetVectorIntersection(track1,track2);

		int num_isect = (int) isect.size();

		if (num_isect == 0)
			return;
    
		MatchIndex idx = GetMatchIndex(img1, img2);

		std::vector<KeypointMatch> &matches = m_matches_table.GetMatchList(idx); 

		matches.clear();
		matches.resize(num_isect);

		for (int i = 0; i < num_isect; i++) 
		{
			 int tr = isect[i];
#if 0
			int num_views = (int) m_track_data[tr].m_views.size();
			int k1 = -1, k2 = -1;

			for (int j = 0; j < num_views; j++) {
				if (m_track_data[tr].m_views[j].first == img1) {
					k1 = m_track_data[tr].m_views[j].second;
				} 

				if (m_track_data[tr].m_views[j].first == img2) {
					k2 = m_track_data[tr].m_views[j].second;
				} 
			}

			assert(k1 != -1 && k2 != -1);
#endif

			std::pair<std::vector<int>::const_iterator,std::vector<int>::const_iterator> p;

			const std::vector<int> &pt1 = m_image_data[img1].m_visible_points;

			p = equal_range(pt1.begin(), pt1.end(), tr);

			assert(p.first != p.second);
			
			//2. same index‘s position in img1
			int offset = p.first - pt1.begin();			
			int k1 = m_image_data[img1].m_visible_keys[offset];

			const std::vector<int> &pt2 = m_image_data[img2].m_visible_points;

			p = equal_range(pt2.begin(), pt2.end(), tr);
			assert(p.first != p.second);
			//2. same index‘s position in img2
			offset = p.first - pt2.begin();
			int k2 = m_image_data[img2].m_visible_keys[offset];

			// 3.
			matches[i] = KeypointMatch(k1, k2);
		}
	}

	void MatchTracks::SetTracks(int image)
	{
		printf("[SetTracks] Setting tracks for image %d...\n", image);

		ImageData& img_data = m_image_data[image];

		int num_tracks = img_data.m_visible_points.size();

		for(int i=0; i < num_tracks; i++)
		{
			int tr = img_data.m_visible_points[i];
			int key = img_data.m_visible_keys[i];

			 assert(key < (int) img_data.m_keys.size());

			 img_data.m_keys[key].m_track = tr;
		}
	}

	cv::Mat MatchTracks::drawImageMatches(int img1, int img2)
	{
		int max_index = img1 > img2 ? img1 : img2;
		int min_index = img1 < img2 ? img1 : img2;

		cv::Mat image_l = m_images[min_index];
		cv::Mat image_r = m_images[max_index];

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

		std::vector<Keypoint> keypoint_l = m_image_data[min_index].m_keys;
		std::vector<Keypoint> keypoint_r = m_image_data[max_index].m_keys;

		MatchIndex idx = GetMatchIndex(min_index,max_index);

		SetMatchesFromTracks(min_index, max_index);

		std::vector<KeypointMatch>& list = m_matches_table.GetMatchList(idx);

		for(int i=0; i < list.size(); i++)
		{
			cv::Point2f pt1 = cv::Point2f(keypoint_l[list[i].m_idx1].m_x,keypoint_l[list[i].m_idx1].m_y);
			cv::Point2f pt2 = cv::Point2f(keypoint_r[list[i].m_idx2].m_x,keypoint_r[list[i].m_idx2].m_y);

			pt2.x += image_l.cols;

			cv::line(drawImage,pt1,pt2,cv::Scalar(0,0,255),1);
		}

		m_matches_table.ClearMatch(idx);

		return drawImage;
	}

	void MatchTracks::ComputeGeometricConstraints(bool overwrite, 
				     int new_image_start)
	{
		int num_images = m_num_images;
		
// 		const char *filename = "constraints.txt";
// 		if (!overwrite)
// 		{
// 			
// 			FILE *f = fopen(filename, "r");
// 			if (f != NULL)
// 			{
// 				ReadGeometricConstraints(filename);
// 				fclose(f);
// 
// 				return;
// 			}
// 		}
		CreateMatchTable();

		if(num_images < 40000)
			WriteMatchTable(".prune");

		//ComputeEpipolarGeometry(new_image_start);
		ComputeTransform();

		MakeMatchListsSymmetric();

		if (num_images < 40000)
            WriteMatchTable(".ransac");

		ComputeTracks(new_image_start);

// set MatchTable with tracks

#if 1
		/* Set match flags */
		int num_tracks = (int) m_track_data.size();
		for (int i = 0; i < num_tracks; i++) {
			TrackData &track = m_track_data[i];
			int num_views = (int) track.m_views.size();

			for (int j = 0; j < num_views; j++) {
				int img1 = track.m_views[j].first;

				assert(img1 >= 0 && img1 < num_images);

				for (int k = j+1; k < num_views; k++) {
					int img2 = track.m_views[k].first;

					assert(img2 >= 0 && img2 < num_images);

					SetMatch(img1, img2);
					SetMatch(img2, img1);
				}
			}
		}
#endif

//         /* Set match flags */
//         int num_tracks = (int) m_track_data.size();
//         for (int i = 0; i < num_tracks; i++) {
//             TrackData &track = m_track_data[i];
//             int num_views = (int) track.m_views.size();
// 
//             for (int j = 0; j < num_views - 1; j++) {
//                 int img1 = track.m_views[j].first;
// 				int img_index1 = track.m_views[j].second;
// 
//                 assert(img1 >= 0 && img1 < num_images);
// 
//                 for (int k = j+1; k < num_views; k++) {
//                     int img2 = track.m_views[k].first;
// 					int img_index2 = track.m_views[k].second;
// 
//                     assert(img2 >= 0 && img2 < num_images);
//                     
//                     m_matches_table.SetMatch(GetMatchIndex(img1,img2));
// 					m_matches_table.AddMatch(GetMatchIndex(img1,img2),KeypointMatch(img_index1,img_index2));
//                     m_matches_table.SetMatch(GetMatchIndex(img2,img1));
// 					m_matches_table.AddMatch(GetMatchIndex(img2,img1),KeypointMatch(img_index2,img_index1));
//                 }
//             }
//         }

		if(num_images < 40000)
            WriteMatchTable(".corresp");

		//WriteGeometricConstraints(filename);

	}

	void MatchTracks::WriteGeometricConstraints(const char *filename)
	{
		FILE *f = fopen(filename, "w");
		if (f == NULL) {
			printf("Error opening file %s for writing\n", filename);
			return;
		}
		unsigned int num_images = m_num_images;
		fprintf(f, "%d\n", num_images);
		
		/* Count the number of transforms to write */
		unsigned long long num_transforms = 0;
		for (unsigned int i = 0; i < num_images; i++)
		{

			MatchAdjList &nbrs = m_matches_table.GetNeighbors(i);
			int num_nbrs = (int) nbrs.size();
		
			MatchAdjList::iterator iter;
			for (iter = nbrs.begin(); iter != nbrs.end(); iter++)
			{
				int j = iter->m_index; // *iter; // nbrs[nbr];

				MatchIndex idx = GetMatchIndex(i, j);
				if (m_transforms.find(idx) != m_transforms.end()) 
					num_transforms++;
			}
		}

		printf("[WriteGeometricConstraints] Writing %llu transforms\n", 
			   num_transforms);
		fprintf(f, "%llu\n", num_transforms);

		for (unsigned int i = 0; i < num_images; i++) 
		{
			MatchAdjList &nbrs = m_matches_table.GetNeighbors(i);
			MatchAdjList::iterator iter;
			
			for (iter = nbrs.begin(); iter != nbrs.end(); iter++)
			{
				unsigned int j = iter->m_index; // *iter; // nbrs[nbr];

				MatchIndex idx = GetMatchIndex(i, j);

				if (m_transforms.find(idx) != m_transforms.end()) 
				{
					fprintf(f, "%d %d\n", i, j);
					m_transforms[idx].WriteToFile(f);
					 fprintf(f, "0\n");
				}

			} // for iter

		} // for i

		/* Write the tracks */
		int num_tracks = (int) m_track_data.size();
		fprintf(f, "%d\n", num_tracks);
		for (int i = 0; i < num_tracks; i++)
		{
			m_track_data[i].Write(f);
		}
    
		fclose(f);

	}

	void MatchTracks::ReadGeometricConstraints(const char *filename)
	{
		FILE *f = fopen(filename, "r");

		int num_images;

		fscanf(f, "%d\n", &num_images);
		if (num_images != m_num_images) 
		{
			printf("[ReadGeometricConstraints] Error: number of images don't match!\n");
			return;
		}

		m_transforms.clear();

		RemoveAllMatches();
		unsigned long long num_transforms;
		fscanf(f, "%llu\n", &num_transforms);

		for (unsigned long int count = 0; count < num_transforms; count++)
		{
			int i, j;
			fscanf(f, "%d %d\n", &i, &j);

			//printf("[ReadTransforms] [%d,%d]\n",i,j);

			MatchIndex idx = GetMatchIndex(i, j);

			m_matches_table.SetMatch(idx);
			m_transforms[idx] = TransformInfo();

			/* Read the transform information */
			m_transforms[idx].ReadFromFile(f);

			int flag;
			fscanf(f, "%d\n",flag);
		}
		
		/* Read the tracks */
		int num_tracks = 0;
		fscanf(f, "%d", &num_tracks);

		printf("[ReadGeometricConstraints] Reading %d tracks\n", num_tracks);

		m_track_data.clear();

		int ncount = 0;
		for (int i = 0; i < num_tracks; i++)
		{
			TrackData track;
			track.Read(f);

			int num_views = (int) track.m_views.size();

			if (num_views < m_min_track_views)
            continue;

			if (num_views > m_max_track_views)
				continue;

			for (int j = 0; j < num_views; j++)
			{
				int img = track.m_views[j].first;
				int key = track.m_views[j].second;

				m_image_data[img].m_visible_points.push_back(ncount);
				m_image_data[img].m_visible_keys.push_back(key);
			}

			m_track_data.push_back(track);
			ncount++;
		}

		fclose(f);


		/*int i = 1;
		{
			std::vector<int> tracks = m_image_data[i].m_visible_points;
			std::vector<int> keys  = m_image_data[i].m_visible_keys;
			std::cout<<"image tracks "<<i<<" : ";
			for(int j = 0; j <tracks.size(); j++ )
			{
				printf("[%d]: %d",j,tracks[j]);
				if(j % 500 == 0)
					std::cout<<std::endl;
			}
			std::cout<<std::endl;
			std::cout<<"image keys "<<i<<" : ";
			for(int j = 0; j <tracks.size(); j++ )
			{
				printf("[%d]: %d",j,keys[j]);
				if(j % 500 == 0)
					std::cout<<std::endl;
			}
			std::cout<<std::endl;
		}*/

	}

	void MatchTracks::WriteMatchTable(const char *append)
	{
		int num_images = m_num_images;

		char buf[256];
		sprintf(buf, "nmatches%s.txt", append);
		FILE *f0 = fopen(buf, "w");

		sprintf(buf, "matches%s.txt", append);
		FILE *f1 = fopen(buf, "w");
    
		if (f0 == NULL || f1 == NULL) {
			printf("[WriteMatchTable] "
				   "Error opening files for writing.\n");
			return;
		}

		fprintf(f0, "%d\n", num_images);

		for (int i = 0; i < num_images; i++) {
			for (int j = 0; j < num_images; j++) {
				if (i >= j) {
					fprintf(f0, "0 ");
					fprintf(f1, "\n");
				} else {
					if (m_matches_table.Contains(GetMatchIndex(i, j))) 
					{
						MatchIndex idx = GetMatchIndex(i, j);
						std::vector<KeypointMatch> &list = 
							m_matches_table.GetMatchList(idx);

						unsigned int num_matches = list.size();

						fprintf(f0, "%d ", num_matches);
                    
						for (unsigned int k = 0; k < num_matches; k++) {
							KeypointMatch m = list[k];
							fprintf(f1, "%d %d ", m.m_idx1, m.m_idx2);    
						}
						fprintf(f1, "\n");
					} else {
						fprintf(f0, "0 ");
					}
				}
			}

			fprintf(f0, "\n");
		}

		fclose(f0);
		fclose(f1);
	}

	/* Compute epipolar geometry between all matching images */
	void MatchTracks::ComputeEpipolarGeometry(int new_image_start)
	{
		 m_transforms.clear();

		 unsigned int num_images = m_num_images;

		 std::vector<MatchIndex> remove;

		for (unsigned int i = 0; i < num_images; i++) 
		{
			MatchAdjList::iterator iter;
			for (iter = m_matches_table.Begin(i); iter != m_matches_table.End(i); iter++) 
			{
				int j = iter->m_index; // first;
				
				assert(ImagesMatch(i, j));
				MatchIndex idx = GetMatchIndex(i, j);
				MatchIndex idx_rev = GetMatchIndex(j, i);

				bool connect12 = ComputeEpipolarGeometry(i, j);

				if (!connect12)
				{
					// RemoveMatch(i, j);
					// RemoveMatch(j, i);
					remove.push_back(idx);
					remove.push_back(idx_rev);

					m_transforms.erase(idx);
					m_transforms.erase(idx_rev);
				}
			}//iter
		}//i

		int num_removed = (int) remove.size();

		for (int i = 0; i < num_removed; i++) {
			int img1 = remove[i].first;
			int img2 = remove[i].second;

			// RemoveMatch(img1, img2);
			m_matches_table.RemoveMatch(GetMatchIndex(img1, img2));
		}
	}

	bool MatchTracks::ComputeEpipolarGeometry(int idx1, int idx2 ) 
	{

		MatchIndex offset = GetMatchIndex(idx1, idx2);
		MatchIndex offset_rev = GetMatchIndex(idx2, idx1);

		m_transforms[offset] = TransformInfo();
        m_transforms[offset_rev] = TransformInfo();

		// make sure the Mat F type( CV_64F )  and whether F.t() equal to F.inv() ?
		cv::Mat F = m_fmatrix[offset];
		if(F.empty())
			return false;

		cv::Mat Finv = F.t();

		memcpy(m_transforms[offset].m_fmatrix,F.data,9 * sizeof(double));
		memcpy(m_transforms[offset_rev].m_fmatrix,Finv.data,9 * sizeof(double));

		return true;
	}

	bool three_random_correspondences(int match_number,int * n1, int * n2, int * n3)
	{
		int shot = 0;
		*n1 = rand() % match_number;	
		//////////////////////////////////////////////////////////////////////////
		do
		{
			*n2 = rand() % match_number;
			shot++; if (shot > 100) return false;
		}
		while(*n2 == *n1);
		//////////////////////////////////////////////////////////////////////////
		shot = 0;
		do
		{
			*n3 = rand() % match_number;
			shot++; if (shot > 100) return false;
		}
		while(*n3 == *n1 || *n3 == *n2);

		return true;
	}
	
	bool estimate_horn(cv::Point2f u1, cv::Point2f u2, cv::Point2f u3, 
		cv::Point2f v1, cv::Point2f v2, cv::Point2f v3, cv::Mat & dst)
	{
		double a[6][6]= { {u1.x,u1.y, 1.0,    0,    0,   0},
		{   0,   0,   0, u1.x, u1.y, 1.0},
		{u2.x,u2.y, 1.0,    0,    0,   0},
		{   0,   0,   0, u2.x, u2.y, 1.0},
		{u3.x,u3.y, 1.0,    0,    0,   0},
		{   0,   0,   0, u3.x, u3.y, 1.0}
		};

		double b[6] = {v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};

		cv::Mat AA = cv::Mat(6,6,CV_64FC1,a);
		cv::Mat B = cv::Mat(6,1,CV_64FC1,b);
		//	cout<<B;

		cv::Mat X(6,1,CV_64FC1);

		bool ok = cv::solve(AA,B,X,cv::DECOMP_SVD);

		if (!ok)
		{
			return false;
		}

		assert(3 == dst.cols && dst.rows == 3);

		dst.at<double>(0,0) = X.ptr<double>(0)[0];
		dst.at<double>(0,1) = X.ptr<double>(1)[0];
		dst.at<double>(0,2) = X.ptr<double>(2)[0];
		dst.at<double>(1,0) = X.ptr<double>(3)[0];
		dst.at<double>(1,1) = X.ptr<double>(4)[0];
		dst.at<double>(1,2) = X.ptr<double>(5)[0];

		return true;
	}

	int compute_support_for_transformation(cv::Mat &H ,std::vector<cv::Point2f> object_points, std::vector<cv::Point2f> keypoints,std::vector<bool>& inliers)
	{
		inliers.clear();
		
		double det = (H.ptr<double>(0)[0] * H.ptr<double>(1)[1] - H.ptr<double>(0)[1] * H.ptr<double>(1)[0]);	
		int matcher_size = object_points.size();
		if (det < 0. || det > 4 * 4)
		{
			inliers.resize(matcher_size,false);
			return 0;
		}

		int result = 0;
		for (int i=0; i< matcher_size;i++)
		{
			cv::Point2f dst;
			dst.x = object_points[i].x * H.ptr<double>(0)[0] + object_points[i].y * H.ptr<double>(0)[1] + H.ptr<double>(0)[2];
			dst.y = object_points[i].x * H.ptr<double>(1)[0] + object_points[i].y * H.ptr<double>(1)[1] + H.ptr<double>(1)[2];

			cv::Point2f real_point = keypoints[i];
			if( (dst.x - real_point.x)*(dst.x - real_point.x) + (dst.y - real_point.y)*(dst.y - real_point.y) < RANSAC_DIST_THRESHOLD* RANSAC_DIST_THRESHOLD )
			{
				result++;
				inliers.push_back(true);
			}
			else
				inliers.push_back(false);
		}

		return result;
	}

	cv::Mat estimate_transform(std::vector<cv::Point2f> object_points, std::vector<cv::Point2f> keypoints,std::vector<bool>& inliers, int max_ransac_iterations)
	{
		int match_size = object_points.size();
		int iteration = 0;
		int best_support = -1;

		cv::Mat Best_H;
		
		do 
		{
			int n1,n2,n3;

			if(!three_random_correspondences(match_size,&n1,&n2,&n3))
				break;

			//object_points
			cv::Point2f u1 = object_points[n1];
			cv::Point2f u2 = object_points[n2];
			cv::Point2f u3 = object_points[n3];

			//keypoints
			cv::Point2f v1 = keypoints[n1];
			cv::Point2f v2 = keypoints[n2];
			cv::Point2f v3 = keypoints[n3];

			cv::Mat h(3,3,CV_64FC1);

			h.ptr<double>(2)[0] = 0.0;
			h.ptr<double>(2)[1] = 0.0;
			h.ptr<double>(2)[2] = 1.0;

			//计算仿射变换矩阵H;
			estimate_horn(u1,u2,u3,v1,v2,v3,h);

			std::vector<bool> _inliers;
			int support = compute_support_for_transformation(h,object_points,keypoints,_inliers);

			if( support > best_support)
			{
				best_support = support;
				Best_H = h.clone();
				if (support > ransac_stop_support)
					break;
			}

		}while(iteration++ < max_ransac_iterations);

		if(Best_H.empty())
			return Best_H;

		int support = compute_support_for_transformation(Best_H,object_points,keypoints,inliers);

		return Best_H;

	}

	std::vector<int> MatchTracks::EstimateTransform(int idx1,int idx2, double M[])
	{
		MatchIndex offset = GetMatchIndex(idx1, idx2);

		std::vector<KeypointMatch> &list = m_matches_table.GetMatchList(offset);

		std::vector<bool> inliers;
		std::vector<cv::Point2f> pt1,pt2;

		for(int i = 0; i < list.size(); i++)
		{
			int ptidx1 = list[i].m_idx1;
			int ptidx2 = list[i].m_idx2;

			pt1.push_back(cv::Point2f(m_image_data[idx1].m_keys[ptidx1].m_x,m_image_data[idx1].m_keys[ptidx1].m_y));
			pt2.push_back(cv::Point2f(m_image_data[idx2].m_keys[ptidx2].m_x,m_image_data[idx2].m_keys[ptidx2].m_y));
		}

		cv::Mat H;
#ifdef USING_OPENCV
		std::vector<unsigned char> match_mask;
		H = cv::findHomography(pt1,pt2,CV_RANSAC,3.0,match_mask);
		for (int i = 0; i < match_mask.size(); i++)
		{
			bool vd = match_mask[i] != 0 ? true : false;
			inliers.push_back(vd);
		}

		std::vector<cv::Point2f> per_pt;
		perspectiveTransform(pt1,per_pt,H);

		double projErr = 0.0;
		int inlier_num = 0;
		for (int i = 0; i < match_mask.size(); i++)
		{
			if((int)match_mask[i] == 0 ) continue;

			double dist = sqrt((per_pt[i].x - pt2[i].x)*(per_pt[i].x - pt2[i].x) + (per_pt[i].y - pt2[i].y)*(per_pt[i].y - pt2[i].y));

			projErr += dist;
			inlier_num++;
		}
		if(inlier_num > 0)
			projErr /= (inlier_num);

		//std::cout<<"re-projection error: "<<projErr<<std::endl;
#else
		H = estimate_transform(pt1,pt2,inliers,1000);

		std::vector<cv::Point2f> per_pt;
		perspectiveTransform(pt1,per_pt,H);

		double projErr = 0.0;
		int inlier_num = 0;
		for (int i = 0; i < inliers.size(); i++)
		{
			if(!inliers[i]) continue;
			double dist = sqrt((per_pt[i].x - pt2[i].x)*(per_pt[i].x - pt2[i].x) + 
				(per_pt[i].y - pt2[i].y)*(per_pt[i].y - pt2[i].y));

			projErr += dist;
			inlier_num++;
		}

		if(inlier_num > 0)
			projErr /= (inlier_num);

		std::cout<<"re-projection error: "<<projErr<<std::endl;
#endif // USING_OPENCV

		if(!H.empty())
		{
			memcpy(M,H.data,9 * sizeof(double));
			//std::cout<<"[EstimateTransform] "<<H<<std::endl;
		}

		std::vector<int> inlierIndex;
		for (int i = 0; i < inliers.size(); i ++)
		{
			if(!inliers[i]) continue;
			inlierIndex.push_back(i);
		}

		return inlierIndex;
	}

	void MatchTracks::ComputeTransform()
	{
		unsigned int num_images = m_num_images;

		m_transforms.clear();

		for (unsigned int i = 0; i < num_images; i++) {
			MatchAdjList::iterator iter;
			for (iter = m_matches_table.Begin(i); iter != m_matches_table.End(i); iter++) {
				unsigned int j = iter->m_index;

				assert(ImagesMatch(i, j));

				MatchIndex idx = GetMatchIndex(i, j);
				MatchIndex idx_rev = GetMatchIndex(j, i);

				m_transforms[idx] = TransformInfo();
				m_transforms[idx_rev] = TransformInfo();

				bool connect12 = ComputeTransform(i, j);

				if (!connect12) {
					m_matches_table.RemoveMatch(idx);
					m_matches_table.RemoveMatch(idx_rev);
					// m_match_lists.erase(idx);

					m_transforms.erase(idx);
					m_transforms.erase(idx_rev);
				}
			} // iter
		} // i

		/* Print the inlier ratios */
		FILE *f = fopen("pairwise_scores.txt", "w");

		for (unsigned int i = 0; i < num_images; i++) {
			MatchAdjList::iterator iter;

			for (iter = m_matches_table.Begin(i); iter != m_matches_table.End(i); iter++) {
				unsigned int j = iter->m_index; // first;

				assert(ImagesMatch(i, j));

				// MatchIndex idx = *iter;
				MatchIndex idx = GetMatchIndex(i, j);
				fprintf(f, "%d %d %0.5f\n", i, j, 
					m_transforms[idx].m_inlier_ratio);
			}
		}

		fclose(f);
	}

	bool MatchTracks::ComputeTransform(int idx1,int idx2)
	{
		if(idx1 == idx2)
		{
			printf("[ComputeTransform] Error: computing tranform "
				"for identical images\n");
			return false;
		}
		
		double M[9];
		std::vector<int> inliers = EstimateTransform(idx1,idx2,M);

		int num_inliers = (int) inliers.size();

		MatchIndex offset = GetMatchIndex(idx1, idx2);
		MatchIndex offset_rev = GetMatchIndex(idx2, idx1);
		std::vector<KeypointMatch> &list = m_matches_table.GetMatchList(offset);
		
// 		printf("Inliers[%d,%d] = %d out of %d\n", idx1, idx2, num_inliers, 
// 			(int) list.size());

#define MIN_INLIERS 10
		if (num_inliers >= MIN_INLIERS)
		{
			m_transforms[offset].m_num_inliers  = num_inliers;
			m_transforms[offset].m_inlier_ratio = 
				((double) num_inliers) / ((double) list.size());

			cv::Mat_<double> homo = (cv::Mat_<double>(3, 3)<< M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8]);
			cv::Mat_<double> hinv = homo.inv();

			memcpy(m_transforms[offset].m_H, M, 9 * sizeof(double));
			memcpy(m_transforms[offset_rev].m_H, hinv.data, 9 * sizeof(double));
#if 0
			printf("Ratio[%d,%d] = %0.3e\n", 
				idx1, idx2, m_transforms[offset].m_inlier_ratio);
			
			int i, j;
			for (i = 0; i < 3; i++) {
				printf("  ");
				for (j = 0; j < 3; j++) {
					printf(" %0.6e ", M[i * 3 + j]);
				}
				printf("\n");
			}
			printf("\n");
#endif

			return true;
		}
		else
			return false;
	}

	MatchTracks::~MatchTracks(void)
	{

	}

}
