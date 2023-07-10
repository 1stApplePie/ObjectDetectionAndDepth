/**
 * @file ClusterLidarPoints.hpp
 * @author Stella Park
 * @author Chanwoo Lee
 * @author Jaejin Kim
 * @brief ClusterLidarPoints class header file
 * @version 0.5
 * @date 2023-06-29
 */

#include <numeric>

#include "ObjectDetectionSystem/ClusterLidarPoints.hpp"

namespace Xycar {

template <typename PREC>
void ClusterLidarPoints<PREC>::setConfiguration(const YAML::Node& config)
{
    mEps = config["DBSCAN"]["epsilon"].as<PREC>();
    mMnPts = config["DBSCAN"]["minimum_points"].as<int32_t>();
}

template <typename PREC>
void ClusterLidarPoints<PREC>::setDB(const std::vector<cv::Point>& DB)
{
    mDB = DB;
}

template <typename PREC>
PREC ClusterLidarPoints<PREC>::calculateDistance(const cv::Point& pt1, const cv::Point& pt2)
{
    return std::sqrt(std::pow(pt1.x - pt2.x, 2) + std::pow(pt1.y - pt2.y, 2));
}

template <typename PREC>
std::vector<int32_t> ClusterLidarPoints<PREC>::rangeQuery(const cv::Point& pt)
{
    std::vector<int32_t> neighbors;
    for (size_t i = 0; i < mDB.size(); i++) {
        if (calculateDistance(mDB[i], pt) < mEps) neighbors.push_back(i);
    }
    return neighbors;
}

template <typename PREC>
std::vector<int32_t> ClusterLidarPoints<PREC>::setUnion(const std::vector<int32_t>& set1, const std::vector<int32_t>& set2)
{
    std::vector<int32_t> uSet;
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(uSet));
    
    return uSet;
}

template <typename PREC>
std::vector<cv::Point> ClusterLidarPoints<PREC>::calculateCenters()
{
    std::vector<cv::Point> DBCopy = {mDB.begin(), mDB.end()};
    std::vector<int32_t> labelsCopy = {mLabels.begin(), mLabels.end()};

    // below can be substituted with using member cluster counting variable
    // std::sort(labelsCopy.begin(), labelsCopy.end());
    // labelsCopy.erase(std::unique(labelsCopy.begin(), labelsCopy.end(), labelsCopy.end()));
    // int32_t numClusters = labelsCopy.size();

    std::vector<cv::Point> centers;

    for (size_t numC = 1; numC <= mCCnt; numC++) {
        // keeping tmpCluster in case
        std::vector<cv::Point> tmpCluster;
        int32_t sumX = 0;
        int32_t sumY = 0;
        int32_t cntPt = 0;
        for (size_t pt = 0; pt < DBCopy.size(); pt++) {
            if (labelsCopy[pt] == numC) {
                tmpCluster.push_back(DBCopy[pt]);
                cntPt += 1;
                sumX += DBCopy[pt].x;
                sumY += DBCopy[pt].y;
            }
        }
        if (cntPt != 0) {
            cv::Point tmpC = cv::Point(static_cast<int32_t>(sumX/cntPt + 0.5f), static_cast<int32_t>(sumY/cntPt + 0.5f));
            centers.push_back(tmpC);
        }
    }
    return centers;
}

template <typename PREC>
std::pair<std::vector<int32_t>, int32_t> ClusterLidarPoints<PREC>::DBSCAN()
{
    int32_t dbSize = mDB.size();
    int32_t cCnt = 0;
    // 0 == Cluster::UNDEFINED
    std::vector<int32_t> labels(dbSize, 0);

    for (size_t pt = 0; pt < dbSize; pt++) {
        // 0 == Cluster::UNDEFINED
        if (labels[pt] != 0) continue;
        std::vector<int32_t> neighbors = rangeQuery(mDB[pt]);
        if(neighbors.size() < mMnPts) {
            // -1 == Cluster::NOISE;
            labels[pt] = -1;
            continue;
        }
        cCnt += 1;
        labels[pt] = cCnt;

        while (!neighbors.empty()) {
            int32_t q = neighbors.back();
            neighbors.pop_back();
            // if (labels[q] == Cluster::NOISE)
            if (labels[q] == -1) labels[q] = cCnt;
            // if (labels[q] != Cluster::UNDEFINED)
            if (labels[q] != 0) continue;
            labels[q] = cCnt;
            std::vector<int32_t> neighborsOfNeighbor = rangeQuery(mDB[q]);
            if (neighborsOfNeighbor.size() >= mMnPts) {
                neighbors = setUnion(neighbors, neighborsOfNeighbor);
            }
        }
    }

    return {labels, cCnt};
}

template <typename PREC>
std::vector<cv::Point> ClusterLidarPoints<PREC>::getCenters(const std::vector<cv::Point>& lidarPts)
{
    setDB(lidarPts);
    //DEBUGGING DELETE AFTERWARD
    // std::cout << "#############################DEBUGGING at ClusterLidarPoints.cpp line 132-151 debugDBSCAN should be deleted after adjusting epsilon" << std::endl;
    // cv::Mat debugDBSCAN = cv::Mat::zeros(540, 540, CV_8UC3);
    // for (int i = 0; i < mDB.size(); i++) {
    //     cv::circle(debugDBSCAN, cv::Point(mDB[i].x, mDB[i].y), 1, cv::Scalar(128, 0, 0));
    // }

    const auto[labels, cCnt] = DBSCAN();
    mLabels = {labels.begin(), labels.end()};
    mCCnt = cCnt;

    std::vector<cv::Point> centers = calculateCenters();
    if (centers.size() == 0) return std::vector<cv::Point>();
    return centers;
    //DEBUGGING DELETE AFTERWARD
    // for (int i = 0; i < centers.size(); i++) {
    //     cv::circle(debugDBSCAN, cv::Point(centers[i].x, centers[i].y), 4, cv::Scalar(128, 0, 128));
    //     cv::putText(debugDBSCAN, std::to_string(i+1), cv::Point(centers[i].x, centers[i].y), 0, 0.4, cv::Scalar(255, 0, 255));
    // }
    // cv::imshow("DBSCAN", debugDBSCAN);
}

template class ClusterLidarPoints<float>;
template class ClusterLidarPoints<double>;
} // namespace Xycar
