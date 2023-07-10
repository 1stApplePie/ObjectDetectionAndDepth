/**
 * @file ClusterLidarPoints.hpp
 * @author Stella Park
 * @author Chanwoo Lee
 * @author Jaejin Kim
 * @brief ClusterLidarPoints class header file
 * @version 0.5
 * @date 2023-06-29
 */

#ifndef CLUSTER_LIDAR_POINTS_HPP_
#define CLUSTER_LIDAR_POINTS_HPP_

#include <iostream>

#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>

namespace Xycar {

/**
 * @brief Lidar Points status defined is positive value standing for cluster number
 */
enum class Cluster : int8_t
{
    UNDEFINED = 0,
    NOISE = -1
};

/**
 * @brief DBSCANner for Lidar point class
 * @tparam PREC Precision of data
 */
template <typename PREC>
class ClusterLidarPoints final
{
public:
    using Ptr = ClusterLidarPoints*; ///< Pointer type of this class

    /**
     * @brief To cluster discrete Lidar points and find the center of clusters
     *
     * @param[in] config Configuration including parameters for scanner
     */
    ClusterLidarPoints(const YAML::Node& config) { setConfiguration(config); }

    /**
     * @brief Get the Debug Frame object pointer
     * 
     * @return Return frame for debuging
     */
    std::vector<cv::Point> getCenters(const std::vector<cv::Point>& lidarPts);

private:
    /**
     * @brief Set the parameters from config file
     * 
     * @param[in] config Configuration including parameters for detector
     */
    void setConfiguration(const YAML::Node& config);

    /**
     * @brief Set DB as a member variable
     * 
     * @param[in] DB A vector of Lidar Points on a BEV map
     */
    void setDB(const std::vector<cv::Point>& DB);

    /**
     * @brief Calculate the distance between 2 points
     * 
     * @param[in] pt1, pt2 Points in DB
     * @return Distance between 2 points
     */
    PREC calculateDistance(const cv::Point& pt1, const cv::Point& pt2);

    /**
     * @brief Calculate the distance between 2 points
     * 
     * @param[in] pt1, pt2 Points in DB
     * @return Distance between 2 points
     */
    std::vector<int32_t> rangeQuery(const cv::Point& pt);

    /**
     * @brief Packaging of std::set_union
     * 
     * @param[in] set1, set2 Vectors instead of iterators
     * @return Union vector of 2 input vectors
    */
    std::vector<int32_t> setUnion(const std::vector<int32_t>& set1, const std::vector<int32_t>& set2);

    /**
     * @brief DBSCANs Database of points
     * 
     * @return labels for each Point, the number of clusters found
     */
    std::pair<std::vector<int32_t>, int32_t> DBSCAN();

    /**
     * @brief Get the center points of labeled DB points
     * 
     * @return Center point of each label
     */
    std::vector<cv::Point> calculateCenters();

private:
    // setConfig
    PREC mEps;
    int32_t mMnPts;

    // set during runtime
    std::vector<cv::Point> mDB;  ///< Index of a Point in db is an identifier
    std::vector<int32_t> mLabels;///< Index of a label of Point is same as db's
    int32_t mCCnt;

};
} // namespace Xycar
#endif // CLUSTER_LIDAR_POINTS_HPP_
