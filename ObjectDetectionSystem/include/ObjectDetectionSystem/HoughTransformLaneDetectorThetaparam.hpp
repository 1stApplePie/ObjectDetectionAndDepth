// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file HoughTransformLaneDetector.hpp
 * @author Jongrok Lee (lrrghdrh@naver.com)
 * @author Jiho Han
 * @author Haeryong Lim
 * @author Chihyeon Lee
 * @brief Hough Transform Lane Detector class header file
 * @version 1.1
 * @date 2023-05-02
 */
#ifndef HOUGH_TRANSFORM_LANE_DETECTOR_THETA_PARAM_HPP_
#define HOUGH_TRANSFORM_LANE_DETECTOR_THETA_PARAM_HPP_

#include <iostream>

#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>

namespace Xycar {

/**
 * @brief Line direction left or right
 */
enum class Direction : uint8_t
{
    LEFT = 0,  ///< Line direction LEFT
    RIGHT = 1, ///< Line direction RIGHT
};

/**
 * @brief Hough Transform line index
 */
enum HoughIndex : uint8_t
{
    x1 = 0, ///< First point x
    y1 = 1, ///< First point y
    x2 = 2, ///< Second point x
    y2 = 3, ///< Second point y
};

using Line = cv::Vec4i;             ///< Line between two points
// using Line = cv::Vec2f;      ///< rhos and thetas of lines
using Lines = std::vector<Line>;      ///< Vector of Lines
using Indices = std::vector<int32_t>; ///< Indices of lines

/**
 * @brief Hough Transform Lane Detector Class
 * @tparam PREC Precision of data
 */
template <typename PREC>
class HoughTransformLaneDetectorThetaparam final
{
public:
    using Ptr = HoughTransformLaneDetectorThetaparam*; ///< Pointer type of this class


    static constexpr double kHoughRho = 4.0;                  ///< Distance resolution of the accumulator in pixels.
    static constexpr double kHoughTheta = CV_PI / 180.0f;      ///< Angle resolution of the accumulator in radians. If C++20, CV_PI should be replaced with std::numbers::pi
    static constexpr int32_t kDebugLineWidth = 2;             ///< Thickness of lines for debugging
    static constexpr int32_t kDebugRectangleHalfWidth = 5;    ///< Half ot width of rectangle for debugging
    static constexpr int32_t kDebugRectangleStartHeight = 15; ///< Start height of rectangle for debugging
    static constexpr int32_t kDebugRectangleEndHeight = 25;   ///< End height of rectangle for debugging
    static inline const cv::Scalar kRed = { 0, 0, 255 };      ///< Scalar values of Red
    static inline const cv::Scalar kGreen = { 0, 255, 0 };    ///< Scalar values of Green
    static inline const cv::Scalar kBlue = { 255, 0, 0 };     ///< Scalar values of Blue

    /**
     * @brief Construct a new Hough Transform Lane Detector object
     *
     * @param[in] config Configuration including parameters for detector
     */
    HoughTransformLaneDetectorThetaparam(const YAML::Node& config) { setConfiguration(config); }
    
    //void get_signal(bool leftDector, bool rightDector);

    /**
     * @brief Get the Lane Position object
     *
     * @param[in] image Image for searching lane position
     * @return Left x position and Right x position
     */
    std::pair<int32_t, int32_t> getLanePosition(const cv::Mat& image);

    //std::pair<std::vector<int32_t>, std::vector<cv::Point>> getLanePosition(const cv::Mat& image);
    /**
     * @brief Draw the position rectangles on debug image
     *
     * @param[in] leftPositionX Left x position for drawing rectangular
     * @param[in] rightPositionX Right x position for drawing rectangular
     * @param[in] estimatedPositionX Estimated x position from moving average filter
     */
    void drawRectangles(int32_t leftPositionX, int32_t rightPositionX, int32_t estimatedPositionX);

    /**
     * @brief Get the Debug Frame object pointer
     * @return Return frame for debuging
     */
    const cv::Mat& getDebugFrame() const { return mDebugFrame; };
    const cv::Mat& getDebugRoI() const {return mDebugRoI; };
    const cv::Mat& getDebugObj() const {return mDebugObj; };
    int32_t getStopLineY() const {return mSLpos; };

private:
    /**
     * @brief Set the parameters from config file
     * @param[in] config Configuration including parameters for detector
     */
    void setConfiguration(const YAML::Node& config);

    /**
     * @brief Process Image for binarization
     * @param[in] image Frame image
     * @return Binary image matrix of RoI
     */
    cv::Mat processRoI(const cv::Mat& image);

    /**
     * @brief Divide lines into left and right lane's contour
     * @param[in] image Binary image of RoI
     * @return Binary lane contour image of left and right lanes seperately
     */
    //std::pair<std::vector<cv::Mat>, std::vector<cv::Point>> divideLines(const cv::Mat& image);
    std::vector<cv::Mat> divideLines(const cv::Mat& image);

    /**
     * @brief Get the line positions x
     * @param[in] lines Computed all Hough lines
     * @return Average x position
     */
    int32_t getLinePositionX(const cv::Mat& contour, Direction direction);

    /**
     * @brief Get the vaiables of line equation (First-order polynomial equation, y = mx + b)
     *
     * @param[in] contour Lane edge image of RoI, left/right separately
     * @return x coordinate of lane center
     */
    std::pair<PREC, PREC> getLineParameters(const Lines& lines);

    /**
     * @brief Draw the lines on debug image
     *
     * @param[in] lines Left and right lines
     * @param[in] leftLineIndex Left indices among lines
     * @param[in] rightLineIndex Right indices among lines
     */
    void drawLines(const Lines& lines, const Indices& leftLineIndices, const Indices& rightLineIndices);

private:
    int32_t mCannyEdgeLowThreshold;  ///< Low threshold for Canny edge
    int32_t mCannyEdgeHighThreshold; ///< High threshold for Canny edge
    int32_t mHoughThreshold;         ///< Accumulator threshold parameter. Only those lines are returned that get enough votes
    int32_t mHoughMinLineLength;     ///< Minimum line length. Line segments shorter than that are rejected.
    int32_t mHoughMaxLineGap;        ///< Maximum allowed gap between points on the same line to link them.
    PREC mHoughLineSlopeRange;       ///< Slope range to limit Hough lines

    // Image parameters
    int32_t mImageWidth;     ///< The width of the image
    int32_t mImageHeight;    ///< The height of the image
    int32_t mROIStartHeight; ///< The height of the offset for debugging
    int32_t mROIHeight;      ///< Height of ROI

    // Debug Image and flag
    cv::Mat mDebugFrame;     ///< The frame for debugging
    bool mDebugging;         ///< Debugging or not

    // ROI cropped Image
    cv::Mat mDebugRoI;

    // Binary object Image
    cv::Mat mDebugObj;       ///< Image of left/right/stop line marked

    // Lane parameters
    int32_t leftFirstRange;
    int32_t leftLastRange;
    int32_t rightFirstRange;
    int32_t rightLastRange;
    int32_t minArea;
    int32_t maxArea;
    int32_t p3Height;
    int32_t toDiscard;

    int32_t mSLpos;

    // std::pair<cv::Point, cv::Point> previousPoint;
};
} // namespace Xycar
#endif // HOUGH_TRANSFORM_LANE_DETECTOR_THETA_PARAM_HPP_