/**
 * @file AlignCamBEV.hpp
 * @author Stella Park
 * @author Chanwoo Lee
 * @author Jaejin Kim
 * @brief Align points on BEV with Bbox detected in cam image
 * @version 0.5
 * @date 2023-07-04
 */

#ifndef ALIGN_CAM_BEV_HPP
#define ALIGN_CAM_BEV_HPP

#include <iostream>
#include <map>

#include "opencv2/opencv.hpp"

namespace Xycar {

/**
 * @brief DBSCAN for Lidar point class
 * @tparam PREC Precision of data
 */
template <typename PREC>
class AlignCamBEV final
{
public:
    using Ptr = AlignCamBEV*; ///< Pointer type of this class

    /**
     * @brief To align points on BEV with Bbox detected in cam image
     */
    AlignCamBEV() { setVar(); }


    /**
     * @brief Check if cluster centers belongs to the bounding box detected in cam image
     * 
     * @param[in] boxData, cCente 
     * @return bool
     */
    // cv::Mat getBEVmap(const std::vector<cv::Point>& cCenters, const std::vector<std::vector<int32_t>>& boxData, const cv::Mat& img);
    cv::Mat getBEVmap(const std::vector<cv::Point>& cCenters, const std::vector<std::vector<int32_t>>& boxData, const cv::Mat& debugImg);

    /**
     * @brief Check if cluster centers belongs to the bounding box detected in cam image
     * 
     * @param[in] boxData, cCente 
     * @return bool
     */
    std::vector<std::pair<cv::Point, int32_t>> getBEVpts(const std::vector<cv::Point>& cCenters, const std::vector<std::vector<int32_t>>& boxData);

private:
    /**
     * @brief Set the transformation matrix as member variable
     */
    void setVar();

    /**
     * @brief Check if cluster centers belongs to the bounding box detected in cam image
     * 
     * @param[in] boxData, cCente 
     * @return bool
     */
    bool isInRange(const std::vector<int32_t>& boxData, const cv::Point& cCenter);

    std::vector<cv::Point> transferPT(const std::vector<cv::Point>& cCenters);

    std::vector<std::pair<cv::Point, int32_t>> assignCls(const std::vector<cv::Point>& cCsImg, const std::vector<std::vector<int32_t>>& boxData);

private:
    cv::Mat mH;
    std::map<int, cv::Scalar> mColorMap;
};
} // namespace Xycar
#endif // ALIGN_CAM_BEV_HPP