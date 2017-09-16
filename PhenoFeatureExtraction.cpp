#include "PhenoFeatureExtraction.h"


void PhenoFeatureExtraction::smallClusterRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, int clusterTolerance, int minClusterSize, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out)
{

	// Euclidean cluster, remove small clusters
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance(clusterTolerance); //distance mm
	ec.setMinClusterSize(1);
	ec.setMaxClusterSize(cloud_in->points.size());
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_in);
	ec.extract(cluster_indices);

	cloud_out->points.clear();

	for(int j=0; j<cluster_indices.size(); j++)
	{
		if(cluster_indices[j].indices.size() > minClusterSize)
		for(int i=0; i<cluster_indices[j].indices.size(); i++)
			cloud_out->push_back(cloud_in->points[cluster_indices[j].indices[i]]);
	}		
}	

void PhenoFeatureExtraction::computeSlicedSubPointCloudVec(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, int numSubBox, std::string sliceDir)
{
	if(numSubBox <= 0)
		return;

	if(sliceDir != "y" && sliceDir != "x" )
		return;
	
	slicedSubPointCloudVec.clear();

	pcl::getMinMax3D(*cloud_in, _min_point_AABB, _max_point_AABB); 

	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud(cloud_in);
	pass.setFilterFieldName(sliceDir);			
	pass.setFilterLimitsNegative(false);

	float sliceWidth, min, max;

	if(sliceDir == "y")
	{
		sliceWidth = (_max_point_AABB.y - _min_point_AABB.y)/numSubBox; 
		min = _min_point_AABB.y;
		max = _max_point_AABB.y;
	}
	else
	{ 
		sliceWidth = (_max_point_AABB.x - _min_point_AABB.x)/numSubBox; 
		min = _min_point_AABB.x;
		max = _max_point_AABB.x;
	}

	for(int i=0; i<numSubBox; i++)
	{
		// get sub set 
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub(new pcl::PointCloud<pcl::PointXYZRGB>);

		pass.setFilterLimits(min+i*sliceWidth, min+(i+1)*sliceWidth);

		pass.filter(*sub);

		slicedSubPointCloudVec.push_back(sub);
	}

	sliceAxis = sliceDir;
}


void PhenoFeatureExtraction::getConvexHull(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull, 
					   std::vector<pcl::Vertices>& vertices_chull, double& volume, double& area)
{
	pcl::ConvexHull<pcl::PointXYZRGB> chull;
	chull.setDimension(3);
	chull.setComputeAreaVolume(true);
	
	vertices_chull.clear();
	cloud_hull->points.clear();

	chull.setInputCloud (cloud_in);	
	
	chull.reconstruct (*cloud_hull, vertices_chull);
	
	for(int i=0; i<cloud_hull->points.size(); i++)
	{
		cloud_hull->points[i].r = 200;
		cloud_hull->points[i].g = 200;
		cloud_hull->points[i].b = 200;			
	}

	volume = chull.getTotalVolume();
	area = chull.getTotalArea();
}

double PhenoFeatureExtraction::getPlantVolume(double volumeThresh, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& cloudHullVec,
					      std::vector<std::vector<pcl::Vertices>>& verticesChullVec, int & numSubBoxUsed,
				              std::vector<double> & sub_volume_vec, double & sliceVolume, std::vector<double> & sub_area_vec)
{
	double sumVolume = 0.;

	if(slicedSubPointCloudVec.size() <= 0)
		return sumVolume;

	sliceVolume = (_max_point_AABB.x-_min_point_AABB.x)*(_max_point_AABB.y-_min_point_AABB.y)
			     *(_max_point_AABB.z-_min_point_AABB.z)/slicedSubPointCloudVec.size();

	numSubBoxUsed = 0;

	subBoxValidMask.clear();

	subBoxValidMask.resize(slicedSubPointCloudVec.size());

	sub_volume_vec.clear();
	sub_area_vec.clear();

	for(int i=0; i<slicedSubPointCloudVec.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZRGB>);

		std::vector<pcl::Vertices> vertices_chull;
		
		double volume = 0.;

		double area = 0.; 
		
		getConvexHull(slicedSubPointCloudVec[i], cloud_hull, vertices_chull, volume, area);
			
		sub_volume_vec.push_back(volume);
		sub_area_vec.push_back(area);

		//std::cout<<slicedSubPointCloudVec[i]->points.size()<<" "<<volume<<"\n";

		subBoxValidMask[i] = false;

		if(volume > sliceVolume*volumeThresh)
		{
			cloudHullVec.push_back(cloud_hull);

			verticesChullVec.push_back(vertices_chull);

			sumVolume += volume;
			
			numSubBoxUsed++;

			subBoxValidMask[i] = true;		
		}
	}

	return sumVolume;
}

double PhenoFeatureExtraction::getMeshAreaSum(const pcl::PolygonMesh & triangles)
{
	double leafArea = 0.;
	pcl::PointCloud<pcl::PointXYZ> vertices;
	pcl::fromPCLPointCloud2(triangles.cloud, vertices);
	Eigen::Vector3f va, vb;
	
	//go through each triangle
	for(size_t i=0; i<triangles.polygons.size(); ++i)
	{
		va = vertices.points[triangles.polygons[i].vertices[0]].getVector3fMap()-vertices.points[triangles.polygons[i].vertices[1]].getVector3fMap();
		vb = vertices.points[triangles.polygons[i].vertices[0]].getVector3fMap()-vertices.points[triangles.polygons[i].vertices[2]].getVector3fMap();
		leafArea += (va.cross(vb)).norm();
	}

	return leafArea*0.5;
}

void PhenoFeatureExtraction::movingLeastSquareSmooth(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out)
{
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_mls(new pcl::search::KdTree<pcl::PointXYZRGB>);
	cloud_out->points.clear();
	pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
	mls.setComputeNormals(true);
	mls.setInputCloud(cloud_in);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree_mls);
	mls.setSearchRadius(30);
	mls.process(*cloud_out);
}

void PhenoFeatureExtraction::greedyTriangulation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, pcl::PolygonMesh & triangles)
{
	// normal estimation
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree_n->setInputCloud (cloud_in);
	n.setInputCloud (cloud_in);
	n.setSearchMethod (tree_n);
	n.setKSearch (20);
	n.compute (*normals);
	// concatenate XYZRGB and normal
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields (*cloud_in, *normals, *cloud_with_normals);
	// Create search tree
	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree_gp3(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree_gp3->setInputCloud (cloud_with_normals);

	// greedy triangulation
	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
//	pcl::PolygonMesh triangles;
	gp3.setSearchRadius(50);
	gp3.setMu(3);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setMinimumAngle(M_PI/18); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setNormalConsistency(false);
	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree_gp3);
	gp3.reconstruct (triangles);
}

void PhenoFeatureExtraction::projectToAABBSideFace(const pcl::PointXYZRGB & min_point_AABB, const pcl::PointXYZRGB & max_point_AABB,
			   int planeIdx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out)
{
	if(planeIdx<0 || planeIdx>2)
	{
		cloud_out->points.clear();
		return;
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);

	for(int i=0; i<slicedSubPointCloudVec.size(); i++)
		*cloud_in += *slicedSubPointCloudVec[i];
	
	// Create a set of planar coefficients
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);
	coefficients->values[0] = 0;
	coefficients->values[1] = 0;
	coefficients->values[2] = 0;
	
	coefficients->values[planeIdx] = 1.0;

	if(planeIdx == 0)
		coefficients->values[3] = -max_point_AABB.x-500.;
	else if(planeIdx == 1)
		coefficients->values[3] = -max_point_AABB.y/*-500.*/;
	else
		coefficients->values[3] = -max_point_AABB.z-500.;

	// Create the filtering object
	pcl::ProjectInliers<pcl::PointXYZRGB> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(cloud_in);
	proj.setModelCoefficients(coefficients);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
	proj.filter(*pc_tmp);

	// down sample 
	cloud_out->points.clear();
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	vg.setInputCloud(pc_tmp);
	vg.setLeafSize(10.f, 10.f, 10.f);
	vg.filter(*cloud_out);
}

void PhenoFeatureExtraction::weightedMedianHeightWidth(float & weightedMedianHeight, float & weightedMedianWidth)
{
	if(slicedSubPointCloudVec.size() <= 0)
		return;

	if(sliceAxis != "x" && sliceAxis != "y")
		return;

	// break bounding box into several small ones along plant row direction
	float heightCandidates[slicedSubPointCloudVec.size()];
	float heightWeights[slicedSubPointCloudVec.size()];
	float heightWeightSum = 0.f;

	float widthCandidates[slicedSubPointCloudVec.size()];
	float widthWeights[slicedSubPointCloudVec.size()];
	
	for(int i=0; i<slicedSubPointCloudVec.size(); i++)
	{	
		// find the highest point
		pcl::PointXYZRGB tmpMinPoint, tmpMaxPoint;
		pcl::getMinMax3D(*(slicedSubPointCloudVec[i]), tmpMinPoint, tmpMaxPoint); 
		
		heightCandidates[i] = sliceAxis == "y" ? tmpMaxPoint.x : tmpMinPoint.y;
		heightWeights[i] = slicedSubPointCloudVec[i]->points.size();
		heightWeightSum += slicedSubPointCloudVec[i]->points.size();

		widthCandidates[i] = tmpMinPoint.z;
		widthWeights[i] = slicedSubPointCloudVec[i]->points.size();
		//std::cout<<"max Height "<<i<<" "<<heightCandidates[i]<<" size "<<heightWeights[i]<<std::endl;
	}

	// weighted median height
	// selection sort		
	for(int i=1; i<slicedSubPointCloudVec.size(); i++)
	{
		int j = i;
		while(j>0 && heightCandidates[j-1]>heightCandidates[j])
		{
			float tmp = heightCandidates[j-1];
			heightCandidates[j-1] = heightCandidates[j];
			heightCandidates[j] = tmp;
			tmp = heightWeights[j-1];
			heightWeights[j-1] = heightWeights[j];
			heightWeights[j] = tmp;					
			j--;
		}
	}

	// weighted median width
	for(int i=1; i<slicedSubPointCloudVec.size(); i++)
	{
		int j = i;
		while(j>0 && widthCandidates[j-1]>widthCandidates[j])
		{
			float tmp = widthCandidates[j-1];
			widthCandidates[j-1] = widthCandidates[j];
			widthCandidates[j] = tmp;
			tmp = widthWeights[j-1];
			widthWeights[j-1] = widthWeights[j];
			widthWeights[j] = tmp;					
			j--;
		}
	}

	// find 0.5 weight sum
	float medianWeight = 0;
	weightedMedianHeight = 0;
	for(int i=0; i<slicedSubPointCloudVec.size(); i++)
	{
		medianWeight += heightWeights[i]/heightWeightSum;
		if(medianWeight >= 0.5f)
		{
			weightedMedianHeight = heightCandidates[i];
			break;
		}	
	}

	medianWeight = 0;
	weightedMedianWidth = 0;
	for(int i=0; i<slicedSubPointCloudVec.size(); i++)
	{
		medianWeight += widthWeights[i]/heightWeightSum;
		if(medianWeight >= 0.5f)
		{
			weightedMedianWidth = widthCandidates[i];
			break;
		}	
	}
}

void PhenoFeatureExtraction::updateBoundingBox(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloudVec, const pcl::PointXYZRGB & min_point_AABB, const pcl::PointXYZRGB & max_point_AABB, 
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliers)
{
	outliers->points.clear();

	pcl::PassThrough<pcl::PointXYZRGB> pass;

	for(int j=0; j<3; j++)
	{
		float min, max;
		std::string axis;
	
		if(j==0)
		{
			axis = "x";
			min = min_point_AABB.x;
			max = max_point_AABB.x;
		}
		else if(j == 1)
		{
			axis = "y";
			min = min_point_AABB.y;
			max = max_point_AABB.y;
		}
		else
		{
			axis = "z";
			min = min_point_AABB.z;
			max = max_point_AABB.z;
		}

		for(int i=0; i<cloudVec.size(); i++)
		{
			pass.setInputCloud(cloudVec[i]);
			pass.setFilterFieldName(axis);
			pass.setFilterLimits(min, max);

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_inlier(new pcl::PointCloud<pcl::PointXYZRGB>);
			pass.setFilterLimitsNegative(false);
			pass.filter(*pc_inlier);

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_outlier(new pcl::PointCloud<pcl::PointXYZRGB>);
			pass.setFilterLimitsNegative(true);
			pass.filter(*pc_outlier);

			cloudVec[i]->clear();
			*cloudVec[i] += *pc_inlier;

			*outliers += *pc_outlier;	
		}
	}
}

float PhenoFeatureExtraction::refineZMax(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, float initZMax, int iter)
{
	Eigen::Vector4f mass_center;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpInput(new pcl::PointCloud<pcl::PointXYZRGB>);					

	*tmpInput += *cloud_in;

	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setFilterFieldName("z");
	pass.setFilterLimitsNegative(false);

	for(int i=0; i<iter; i++)
	{
		// cut out points beyond mass2_center in z 
		pcl::compute3DCentroid(*tmpInput, mass_center);
		pass.setInputCloud(tmpInput);
		
		pass.setFilterLimits(mass_center(2), initZMax);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);

		pass.filter(*tmp);
		tmpInput->clear();
		*tmpInput += *tmp;
	}

	return mass_center(2);
}

void PhenoFeatureExtraction::compute3LevelHedgeWidths(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const int numSubBox,
							const pcl::PointXYZRGB & min_point_AABB, const pcl::PointXYZRGB & max_point_AABB,
							std::vector<float> & three_hedge_widths) 
{

	if(three_hedge_widths.size() != 3) return;

	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setFilterFieldName("x");
	pass.setFilterLimitsNegative(false);

	float height_step = (max_point_AABB.x - min_point_AABB.x)/3.0f;

	pass.setInputCloud(cloud);

	for(int i=0; i<3; i++)
	{
		std::vector<int> indices;

		pass.setFilterLimits(min_point_AABB.x + i*height_step, min_point_AABB.x + (i+1)*height_step);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr layer_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

		pass.filter(*layer_cloud);

		computeSlicedSubPointCloudVec(layer_cloud, numSubBox, "y");

		float weightedMedianHeight, weightedMedianWidth;

		weightedMedianHeightWidth(weightedMedianHeight, weightedMedianWidth);

		three_hedge_widths[i] = weightedMedianWidth;
	}
}

void PhenoFeatureExtraction::computeMomentOfInertiaAndProjectionOccupancy(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, std::string axis, double & moi, double & occupancy)
{
	if(axis != "z")	return;
	
	moi = 0.;
	
	for(int i=0; i<cloud_in->points.size(); i++)
	{

	}

}
