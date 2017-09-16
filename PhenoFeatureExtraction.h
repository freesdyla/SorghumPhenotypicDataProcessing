#ifndef __PHENO_FEATURE_EXTRACTION_H
#define __PHENO_FEATURE_EXTRACTION_H

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_picking_event.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/fpfh.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/mls.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>

#include <string>

class PhenoFeatureExtraction
{

public:
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> slicedSubPointCloudVec;

	std::vector<bool> subBoxValidMask;

	int numValidSubBox;

	std::string sliceAxis;

	pcl::PointXYZRGB _min_point_AABB, _max_point_AABB;

	void smallClusterRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, int clusterTolerance, int minClusterSize, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out);

	void computeSlicedSubPointCloudVec(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, int numSubBox, std::string sliceDir);

	void getConvexHull(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull, std::vector<pcl::Vertices>& vertices_chull, double& volume, double& area);

	double getPlantVolume( double volumeThresh, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& cloudHullVec,	
	                       std::vector<std::vector<pcl::Vertices>>& verticesChullVec, int & numSubBoxUsed, std::vector<double> & sub_volume_vec, double & slice_volume, 
			       std::vector<double> & sub_area_vec);

	double getMeshAreaSum(const pcl::PolygonMesh & triangles);
	
	void movingLeastSquareSmooth(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out);

	void greedyTriangulation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PolygonMesh & triangles);

	void projectToAABBSideFace(const pcl::PointXYZRGB & min_point_AABB, const pcl::PointXYZRGB & max_point_AABB,
            			   int planeIdx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out);

	void weightedMedianHeightWidth(float & weightedMedianHeight, float & weightedMedianWidth);

	void updateBoundingBox(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloudVec, const pcl::PointXYZRGB & min_point_AABB, const pcl::PointXYZRGB & max_point_AABB, 
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliers);

	float refineZMax(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, float initZMax, int iter);

	void compute3LevelHedgeWidths(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const int numSubBox, const pcl::PointXYZRGB & min_point_AABB, const pcl::PointXYZRGB & max_point_AABB,
							std::vector<float> & three_hedge_widths);

	void computeMomentOfInertiaAndProjectionOccupancy(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, std::string axis, double & moi, double & occupancy);
};

#endif
