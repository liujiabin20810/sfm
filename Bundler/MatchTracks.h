#ifndef  BUNDLER_MATCHTRACKS_HPP__
#define BUNDLER_MATCHTRACKS_HPP__
#pragma once

#include <hash_map>
#include <hash_set>

#include <assert.h>
#include <algorithm>
#include <list>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

typedef std::pair<unsigned long, unsigned long> MatchIndex;
// typedef unsigned long long MatchIndex 

namespace stdext
{
		template<>
		class hash_compare<MatchIndex> {
		public:
			static const size_t bucket_size = 4;
			static const size_t min_buckets = 8;
			size_t
			operator()(const MatchIndex &__x) const
			{ return __x.first * 1529 + __x.second; }

			bool operator()(const MatchIndex &__x1, const MatchIndex &__x2) const {
				return (__x1.first < __x2.first) || (__x1.first == __x2.first && __x1.second < __x2.second);
			}
		};
}

namespace bundler
{

	/* Data struct for matches */
	class KeypointMatch 
	{
		public:
			KeypointMatch():m_idx1(-1), m_idx2(-1)
			{ }
			KeypointMatch(int idx1, int idx2) :	m_idx1(idx1), m_idx2(idx2)
			{ }
			int m_idx1, m_idx2;
	};

//	typedef stdext::hash_map<unsigned int, std::vector<KeypointMatch> >   MatchAdjTable;
	typedef stdext::hash_set<int> HashSetInt;

	class TransformInfo 
	{
	public:

		/* File IO routines */
		void ReadFromFile(FILE *f);
		void WriteToFile(FILE *f);

		/* For object movies */
		double m_fmatrix[9];
		double m_ematrix[9];

		/* For homographies */
		double m_H[9];
		double m_inlier_ratio;
		int m_num_inliers;

		/* For color correction */
		double m_gain[3], m_bias[3];
	};

	typedef stdext::hash_map<MatchIndex, TransformInfo> HashMapTranInfo;

	typedef std::pair<int,int> ImageKey;
	typedef std::vector<ImageKey> ImageKeyVector;

	/* Data for tracks */
	class TrackData 
	{
		public:
			TrackData() : m_extra(-1) {}
			TrackData(ImageKeyVector views) : m_views(views), m_extra(-1) { }

			/* Read/write routines */
			void Read(FILE *f);
			void Write(FILE *f);

			ImageKeyVector m_views;
			int m_extra;
	};

	class Keypoint 
	{
		public:    
		Keypoint()  
		{ m_x = 0.0; m_y = 0.0; m_extra = -1; m_track = -1; 
			m_r = m_g = m_b = 0; }

		Keypoint(float x, float y) :
		m_x(x), m_y(y)
		{ m_extra = -1; m_track = -1; m_r = 0; m_g = 0; m_b = 0; }

		~Keypoint() {}   
    
		float m_x, m_y;              /* Subpixel location of keypoint. */
		// float m_scale, m_ori;     /* Scale and orientation (range [-PI,PI]) */
		unsigned char m_r, m_g, m_b; /* Color of this key */

		int m_extra;  /* 4 bytes of extra storage */
		int m_track;  /* Track index this point corresponds to */
	};

	class ImageData
	{
		public: 
			ImageData(){}

			std::vector<int> m_visible_points;  /* tracks Indices of points visible
                                         * in this image */
			std::vector<int> m_visible_keys;

			std::vector<Keypoint> m_keys;              /* Keypoints in this image */

			//std::vector<int> m_tracks;		/* Track index this point corresponds to */

			std::vector<bool> m_key_flags;			   /* Keypoints used flag */
	};

	class AdjListElem {
		public:
		 bool operator< (const AdjListElem &other) const 
		 {	return m_index < other.m_index;}

		unsigned int m_index;
		 std::vector<KeypointMatch> m_match_list;
	};

	typedef std::vector<AdjListElem> MatchAdjList;

	/* Return the match index of a pair of images */
	MatchIndex GetMatchIndex(int i1, int i2);
	MatchIndex GetMatchIndexUnordered(int i1, int i2);

	class MatchTable
	{
		
	public:

		MatchTable() { }

		MatchTable(int num_images) {
			m_match_lists.resize(num_images);
			// m_neighbors.resize(num_images);
		}

		void SetMatch(MatchIndex idx) { 
			if (Contains(idx))
				return;  // already set

			/* Create a new list */
			// m_match_lists[idx.first][idx.second] = std::vector<KeypointMatch> ();
			// m_match_lists[idx.first].insert(idx.second);
			// std::list<unsigned int> tmp;
			// tmp.push_back(idx.second);
			// m_neighbors[idx.first].merge(tmp);
	#if 0
			MatchAdjList tmp;
			AdjListElem adjlist_elem;
			adjlist_elem.m_index = idx.second;
			tmp.push_back(adjlist_elem);
			m_match_lists[idx.first].merge(tmp);
	#else
			/* Using vector */
			AdjListElem e;
			e.m_index = idx.second;
			MatchAdjList &l = m_match_lists[idx.first];
			MatchAdjList::iterator p = lower_bound(l.begin(), l.end(), e);
			l.insert(p, e);
	#endif
		}

		void AddMatch(MatchIndex idx, KeypointMatch m) {
			assert(Contains(idx));
			// m_match_lists[idx.first][idx.second].push_back(m);
			GetMatchList(idx).push_back(m);
		}

		void ClearMatch(MatchIndex idx) { // but don't erase!
			if (Contains(idx)) {
				// m_match_lists[idx.first][idx.second].clear();
				GetMatchList(idx).clear();
			}
		}
    
		void RemoveMatch(MatchIndex idx) {
			if (Contains(idx)) {
				// m_match_lists[idx.first][idx.second].clear();
				// m_match_lists[idx.first].erase(idx.second);
				std::vector<KeypointMatch> &match_list = GetMatchList(idx);
				match_list.clear();

				// Remove the neighbor
	#if 0
				std::list<unsigned int> &l = m_neighbors[idx.first];
				std::pair<std::list<unsigned int>::iterator,
						  std::list<unsigned int>::iterator> p = 
					equal_range(l.begin(), l.end(), idx.second);

				assert(p.first != l.end());
            
				l.erase(p.first, p.second);
	#endif
				AdjListElem e;
				e.m_index = idx.second;
				MatchAdjList &l = m_match_lists[idx.first];
				std::pair<MatchAdjList::iterator, MatchAdjList::iterator> p = 
					equal_range(l.begin(), l.end(), e);

				assert(p.first != p.second); // l.end());
            
				l.erase(p.first, p.second);        
			}
		}

		unsigned int GetNumMatches(MatchIndex idx) {
			if (!Contains(idx))
				return 0;
        
			// return m_match_lists[idx.first][idx.second].size();
			return GetMatchList(idx).size();
		}

		std::vector<KeypointMatch> &GetMatchList(MatchIndex idx) {
			// assert(Contains(idx));
			// return m_match_lists[idx.first][idx.second];
			if (!Contains(idx))
				return std::vector<KeypointMatch>();

			AdjListElem e;
			e.m_index = idx.second;
			MatchAdjList &l = m_match_lists[idx.first];
			std::pair<MatchAdjList::iterator, MatchAdjList::iterator> p = 
				equal_range(l.begin(), l.end(), e);
    
			assert(p.first != p.second); // l.end());
        
			return (p.first)->m_match_list;
		}
    
		bool Contains(MatchIndex idx) const {
			// return (m_match_lists[idx.first].find(idx.second) != 
			//         m_match_lists[idx.first].end());
			AdjListElem e;
			e.m_index = idx.second;
			const MatchAdjList &l = m_match_lists[idx.first];
			std::pair<MatchAdjList::const_iterator, 
				MatchAdjList::const_iterator> p = 
				equal_range(l.begin(), l.end(), e);
    
			return (p.first != p.second); // l.end());
		}

		void RemoveAll() {
			int num_lists = m_match_lists.size();

			for (int i = 0; i < num_lists; i++) {
				m_match_lists[i].clear();
				// m_neighbors[i].clear();
			}
		}

		unsigned int GetNumNeighbors(unsigned int i) {
			return m_match_lists[i].size();
		}

	#if 0
		std::list<unsigned int> GetNeighbors(unsigned int i) {
			// return m_neighbors[i];
			std::list<unsigned int> nbrs;
			MatchAdjList::iterator p;
			for (p = Begin(i); p != End(i); p++) {
				nbrs.push_back(p->m_index);
			}
			return nbrs;
		}
	#endif

		MatchAdjList &GetNeighbors(unsigned int i) {
			return m_match_lists[i];
		}

		MatchAdjList::iterator Begin(unsigned int i) {
			return m_match_lists[i].begin();
		}
    
		MatchAdjList::iterator End(unsigned int i) {
			return m_match_lists[i].end();
		}
    
	private:
		// std::vector<MatchAdjTable> m_match_lists;
		// std::vector<KeypointMatchList> m_match_lists;
		// std::vector<std::list<unsigned int> > m_neighbors;
		std::vector<MatchAdjList> m_match_lists;
	};

	class MatchTracks
	{
	public:

		void InitMatchTable(std::map<std::pair<int, int>, std::vector<cv::DMatch> > _matches_matrix,
			std::vector<std::vector<cv::KeyPoint>> _images_points ,
			std::map<std::pair<int, int>, cv::Mat > _fmatrix,
			std::vector<cv::Mat> _images,
			int num_images);

		void CreateMatchTable();

		void WriteMatchTable(const char *append);

		/* Compute geometric information about image pairs */
		void ComputeGeometricConstraints(bool overwrite = false,int new_image_start = 0);

		void WriteGeometricConstraints(const char *filename);

		void ReadGeometricConstraints(const char *filename);

		/* Compute epipolar geometry between all matching images */
		void ComputeEpipolarGeometry(int new_image_start);
		
		bool ComputeEpipolarGeometry(int idx1, int idx2) ;

		std::vector<int> EstimateTransform(int idx1,int idx2,double M[]);

		/* Compute rigid transforms between all matching images */
		void ComputeTransform();
		
		bool ComputeTransform(int idx1,int idx2);

		bool ImagesMatch(int i1, int i2);

		void SetMatch(int i1, int i2);

		void RemoveAllMatches();
		/* Prune points that match to multiple targets */
		void PruneDoubleMatches();

		/* Make match lists symmetric */
		void MakeMatchListsSymmetric();

		MatchTracks(void);

		void ComputeTracks(int new_image_start);

		std::vector<int> GetVectorIntersection(const std::vector<int> &v1,
			const std::vector<int> &v2);

		void SetTracks(int image);

		void SetMatchesFromTracks();

		void SetMatchesFromTracks(int img1, int img2);

		cv::Mat drawImageMatches(int img1, int img2);

		~MatchTracks(void);


	public:

		std::map<std::pair<int, int>, std::vector<cv::DMatch> > m_matches_matrix;

		std::map<std::pair<int, int>, cv::Mat > m_fmatrix;

		std::vector<cv::Mat> m_images;

		MatchTable m_matches_table; /* Match matrix table */

		std::vector<TrackData> m_track_data;   /* Information about the
                                            * detected 3D tracks */
		HashMapTranInfo m_transforms;

		std::vector<ImageData> m_image_data; /* Image data */

		int m_num_images;

		/* Number of features matches for
        * an image pair to be considered
        * a match */
		int m_min_num_feat_matches;

		bool m_matches_loaded;    /* Have the matches been loaded? */

		bool m_matches_computed;  /* Have the matches been computed? */

		int m_min_track_views;
        int m_max_track_views;
	};

}
#endif //BUNDLER_MATCHTRACKS_HPP__
