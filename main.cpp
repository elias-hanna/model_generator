#include <iostream>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/extract_clusters.h>

#define USE_KINECT 1
#define KINECT_V2 1

#if USE_KINECT == 1
#if KINECT_V2 == 1
// Kinect2 includes
#define WITH_PCL
#include "libs/libfreenect2pclgrabber/include/k2g.h"
#else
// Kinect1 includes
#include "libs/getter/getter.hpp"
#endif
#endif

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

static int count = 0;
static std::string object_name = "";
static std::string path = "saved_data/";

void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                      void* cloud_void)
{
  PointCloudT::Ptr cloud = *static_cast<PointCloudT::Ptr *>(cloud_void);
  if (event.getKeySym() == "a" && event.keyDown())
  {
    if (mkdir (path.c_str (), 0777) > 0)
    {
      std::cout << "Created directory " << path << "\n";
    }
    if (!object_name.empty ())
    {
      pcl::io::savePCDFile (path + object_name + ".pcd", *cloud);
      std::cout << "'a' was pressed, saving pcd file as " + object_name
                << ".pcd in " << path << " folder.\n";
    }
    else
    {
      pcl::io::savePCDFile(path + "object_" + std::to_string(count) + ".pcd",
                           *cloud);
      std::cout << "'a' was pressed, saving pcd file as object_"
                << count++ << ".pcd in " << path << " folder.\n";
    }
  }
}

int
main (int argc, char** argv)
{
  bool help_asked = pcl::console::find_switch (argc, argv, "-h")
      || pcl::console::find_switch (argc, argv, "--help");

  if(help_asked)
  {
    pcl::console::print_error ("Usage: %s \n "
                               "--name", argv[0]);
    return 1;
  }

  bool name_specified = pcl::console::find_switch (argc, argv, "--name");
  if (name_specified)
    pcl::console::parse (argc, argv, "--name", object_name);

#if USE_KINECT == 1
#if KINECT_V2 == 1
  // Initialize kinect2 getters
  Processor freenectprocessor = OPENGL;
  K2G k2g(freenectprocessor);
  k2g.disableLog ();
#else
  // OpenNIGrabber, used to capture pointclouds from various rgbd cameras
  boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::OpenNIGrabber>();
  // Getter Class to get the point cloud from various capturing devices
  Getter<pcl::PointXYZRGBA> getter( *grabber);#endif
    #endif
    #endif

      PointCloudT::Ptr cloud(new PointCloudT);
  PointCloudT::Ptr cropped_cloud(new PointCloudT);
  PointCloudT::Ptr cropped_and_filtered_cloud(new PointCloudT);
  PointCloudT::Ptr final_cloud(new PointCloudT);

  pcl::PCDWriter writer;
  // Create viewer object
  pcl::visualization::PCLVisualizer::Ptr
      viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->initCameraParameters ();
  viewer->setCameraPosition(0,0,0, 0,0,1, 0,-1,0);
  viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&final_cloud);

  // Planar segmentation
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  // Filtering object
  pcl::ExtractIndices<PointT> extract;
  // Main loop
  while(!viewer->wasStopped ())
  {
    cloud.reset (new PointCloudT);
    // Get the cloud
#if USE_KINECT == 1
#if KINECT_V2 == 1
    copyPointCloud(*k2g.getCloud(), *cloud);
#else
    copyPointCloud(getter.getCloud(), *cloud);
#endif
#endif

    // If a cloud got captured from the device
    if(!cloud->empty())
    {
      float minX = -0.35; float minY = -0.25; float minZ = 0.8;
      float maxX = 0.35; float maxY = 0.5; float maxZ = 1.3;
      pcl::CropBox<PointT> boxFilter;
      boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
      boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
      boxFilter.setInputCloud(cloud);
      boxFilter.filter(*cropped_cloud);

      seg.setInputCloud (cropped_cloud);
      seg.segment (*inliers, *coefficients);

      if (inliers->indices.size () == 0)
      {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
        return (-1);
      }

      // Extract the inliers
      extract.setInputCloud (cropped_cloud);
      extract.setIndices (inliers);
      extract.setNegative (true);
      extract.filter (*cropped_and_filtered_cloud);

      // Creating the KdTree object for the search method of the extraction
      pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
      tree->setInputCloud (cropped_and_filtered_cloud);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<PointT> ec;
      ec.setClusterTolerance (0.02); // 2cm
      ec.setMinClusterSize (100);
      ec.setMaxClusterSize (25000);
      ec.setSearchMethod (tree);
      ec.setInputCloud (cropped_and_filtered_cloud);
      ec.extract (cluster_indices);

      if (cluster_indices.begin () != cluster_indices.end ())
      {
        final_cloud.reset(new PointCloudT);
        std::vector<int>::const_iterator
            pit = cluster_indices.begin ()->indices.begin ();
        for (; pit != cluster_indices.begin ()->indices.end (); ++pit)
        { final_cloud->points.push_back
              (cropped_and_filtered_cloud->points[*pit]); }
        final_cloud->width = final_cloud->points.size ();
        final_cloud->height = 1;
        final_cloud->is_dense = true;

        pcl::visualization::PointCloudColorHandlerRGBAField<PointT>
            rgba(final_cloud);
        if (!viewer->updatePointCloud<PointT> (final_cloud, rgba,
                                               "voxel centroids"))
          viewer->addPointCloud<PointT> (final_cloud, rgba, "voxel centroids");
      }
      else
      {
        pcl::visualization::PointCloudColorHandlerRGBAField<PointT>
            rgba(cloud);
        if (!viewer->updatePointCloud<PointT> (cropped_and_filtered_cloud,
                                               rgba, "voxel centroids"))
          viewer->addPointCloud<PointT> (cropped_and_filtered_cloud,
                                         rgba, "voxel centroids");
      }
      // 60 FPS ?
      viewer->spinOnce (1000/60.f);
    }
  }
  k2g.shutDown();


  return (0);
}
