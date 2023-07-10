// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file HoughTransformLaneDetector.cpp
 * @author Jongrok Lee (lrrghdrh@naver.com)
 * @author Jiho Han
 * @author Haeryong Lim
 * @author Chihyeon Lee
 * @brief hough transform lane detector class source file
 * @version 1.1
 * @date 2023-05-02
 */

#include <numeric>

#include "ObjectDetectionSystem/HoughTransformLaneDetectorThetaparam.hpp"

namespace Xycar {

template <typename PREC>
void HoughTransformLaneDetectorThetaparam<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();
    // mROIStartHeight = config["IMAGE"]["ROI_START_HEIGHT"].as<int32_t>();
    mROIHeight = config["IMAGE"]["ROI_HEIGHT"].as<int32_t>();
    mROIStartHeight = mImageHeight - mROIHeight;
    mCannyEdgeLowThreshold = config["CANNY"]["LOW_THRESHOLD"].as<int32_t>();
    mCannyEdgeHighThreshold = config["CANNY"]["HIGH_THRESHOLD"].as<int32_t>();
    mHoughLineSlopeRange = config["HOUGH"]["ABS_SLOPE_RANGE"].as<PREC>();
    mHoughThreshold = config["HOUGH"]["THRESHOLD"].as<int32_t>();
    mHoughMinLineLength = config["HOUGH"]["MIN_LINE_LENGTH"].as<int32_t>();
    mHoughMaxLineGap = config["HOUGH"]["MAX_LINE_GAP"].as<int32_t>();
    mDebugging = config["DEBUG"].as<bool>();
    leftFirstRange = config["Lane"]["left_idx_first"].as<int32_t>();
    leftLastRange = config["Lane"]["left_idx_last"].as<int32_t>();
    rightFirstRange = config["Lane"]["right_idx_first"].as<int32_t>();
    rightLastRange = config["Lane"]["right_idx_last"].as<int32_t>();
    minArea = config["Lane"]["min_p4"].as<int32_t>();
    maxArea = config["Lane"]["max_p4"].as<int32_t>();
    p3Height = config["Lane"]["p3"].as<int32_t>();
    toDiscard = config["Lane"]["discard"].as<int32_t>();    
}

template <typename PREC>
std::pair<PREC, PREC> HoughTransformLaneDetectorThetaparam<PREC>::getLineParameters(const Lines& lines)
{
    // uint32_t numLines = static_cast<uint32_t>(lines.size());
    // if (numLines == 0)
    //     return { 0.0f, 0.0f };

    // PREC rhoSum = 0.0f;
    // PREC thetaSum = 0.0f;
    // for (int i = 0; i < numLines; i++)
    // {
    //     float rh = lines[i][0];
    //     float th = lines[i][1];
    //     rhoSum += pow(rh, 2);
    //     // rhoSum += rh;
    //     thetaSum += th;
    // }

    // PREC rho = cv::sqrt(rhoSum / numLines);
    // // PREC rho = rhoSum /numLines;
    // PREC theta = thetaSum /numLines;

    // return { rho, theta };

    uint32_t numLines = static_cast<uint32_t>(lines.size());
    if (numLines == 0) {
        std::cout << "lines are not detected" << std::endl;
        return { 0.0f, 0.0f };
    }
    else {
        int32_t xSum = 0;
        int32_t ySum = 0;
        PREC mSum = 0.0f;

        for (int i = 0; i < numLines; i++)
        {
            int32_t x1 = lines[i][HoughIndex::x1];
            int32_t y1 = lines[i][HoughIndex::y1];
            int32_t x2 = lines[i][HoughIndex::x2];
            int32_t y2 = lines[i][HoughIndex::y2];
            xSum += x1 + x2;
            ySum += y1 + y2;
            mSum += static_cast<PREC>((y2 - y1)) / (x2 - x1);
        }

        PREC xAverage = static_cast<PREC>(xSum) / (numLines * 2);
        PREC yAverage = static_cast<PREC>(ySum) / (numLines * 2);
        PREC m = mSum / numLines;
        PREC b = yAverage - m * xAverage;

        //std::cout << "mSum: " << mSum << ", numLines: " << numLines << std::endl;
        //std::cout << "m : " << m << "b : "<< b << std::endl;

        return { m, b };
    }
}

template <typename PREC>
cv::Mat HoughTransformLaneDetectorThetaparam<PREC>::processRoI(const cv::Mat& image)
{
    // ## DEBUGGING masking check ##
    // cv::imshow("before masking", image); 
    cv::Mat maskedImage;
    cv::Mat mask_image = cv::imread("/home/nvidia/xycar_ws/src/ObjectDetectionSystem/src/ObjectDetectionSystem/mask_image.png", cv::IMREAD_GRAYSCALE);

    // cv::imshow("after masking", maskedImage); 

    // channel 3 -> 1
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // filterring
    cv::Mat filteredImage;
    cv::GaussianBlur(grayImage, filteredImage, cv::Size(), 1.0);

    // RoI cropping
    cv::Mat croppedGray = filteredImage(cv::Rect(0, mROIStartHeight, mImageWidth, mROIHeight));
    cv::Mat roiMasking = mask_image(cv::Rect(0, mROIStartHeight, mImageWidth, mROIHeight));

    // average luminance for adaptive thresholding
    cv::Scalar avgL = cv::mean(croppedGray);
    if (mDebugging) std::cout << "average luminance: " << avgL << std::endl;
    int32_t currentTH;
    // if (false) currentTH = mCannyEdgeLowThreshold - std::round((175 - std::round(avgL.val[0]))*0.5f);
    if (avgL.val[0] < 160.f) currentTH = mCannyEdgeLowThreshold - std::round((175 - std::round(avgL.val[0]))*0.5f);
    else currentTH = mCannyEdgeLowThreshold;
    if (mDebugging) std::cout << "adapted threshold: " << currentTH << std::endl;

    // from here is customized
    cv::Mat edgeImage;
    // double th = cv::threshold(croppedGray, edgeImage, mCannyEdgeLowThreshold, mCannyEdgeHighThreshold, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::threshold(croppedGray, edgeImage, currentTH, mCannyEdgeHighThreshold, cv::THRESH_BINARY_INV);
    // cv::adaptiveThreshold(croppedGray, edgeImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 9, 5);
    
    
    cv::subtract(edgeImage, roiMasking, edgeImage);
    
    //cv::Mat closedEdge;
    //cv::morphologyEx(edgeImage, closedEdge, cv::MORPH_CLOSE, cv::Mat());

    // defining objects
    cv::Mat objs, stats, centroids;
    int32_t objCnt = cv::connectedComponentsWithStats(edgeImage, objs, stats, centroids);

    if (mDebugging) {
        edgeImage.copyTo(mDebugRoI);
        cv::imshow("edgeImage", edgeImage);
    }
    return edgeImage;
}

template <typename PREC>
int32_t HoughTransformLaneDetectorThetaparam<PREC>::getLinePositionX(const cv::Mat& contour, Direction direction)
{
    // Lines lines;
    // // cv::HoughLines(roi, allLines, kHoughRho, kHoughTheta, mHoughThreshold, mHoughMinLineLength, mHoughMaxLineGap);
    // cv::HoughLines(contour, lines, 1, kHoughTheta, 15, 0, 0);

    // const auto [rho, theta] = getLineParameters(lines);
    
    // if (std::abs(theta) <= std::numeric_limits<PREC>::epsilon() && std::abs(rho) <= std::numeric_limits<PREC>::epsilon())
    // {
    //     if (direction == Direction::LEFT)
    //         return 0.0f;
    //     else if (direction == Direction::RIGHT)
    //         return static_cast<PREC>(mImageWidth);
    // }

    Lines lines;
    cv::HoughLinesP(contour, lines, kHoughRho, kHoughTheta, mHoughThreshold, mHoughMinLineLength, mHoughMaxLineGap);

    // std::cout << "binarization check: " << cv::countNonZero(contour) << std::endl;
    const auto [m, b] = getLineParameters(lines);
    
    // ## DEBUGGING ##
    // std::cout << m << std::endl;
    // std::cout << b << std::endl;
    // std::cout << std::typeof(m) << std::endl;
    // std::cout << std::typeof(b) << std::endl;

    if (m==0.0f && b==0.0f) {
        // std::cout << "passed" << std::endl;
        // std::cout << "the cause is binarization: " << cv::countNonZero(contour) << std::endl;
        if (direction == Direction::LEFT)
            return 0;
        else if (direction == Direction::RIGHT)
            return mImageWidth;
    } else {
        // std::cout << "nonzero should be: " << cv::countNonZero(contour) << std::endl;
        PREC y = static_cast<PREC>(mROIHeight) * 0.5f;
        int32_t xTmp = std::round((y - b) / m);
        if (xTmp < 0) {
            if (direction == Direction::LEFT)
                return 0;
            else if (direction == Direction::RIGHT)
                return mImageWidth;
        }
        else 
            return xTmp;
    }
    // // ## DEBUGGING rpos is weird
    // if (direction==Direction::RIGHT && mDebugging) {
    //     cv::Mat debuggingRightPos = cv::Mat::zeros(contour.size(), CV_8UC3);
    //     for (size_t i=0; i < lines.size(); i++) {
    //         cv::Point pt1, pt2;
    //         double  a = std::cos(theta), b = std::sin(theta);
    //         double x0 = a*rho, y0 = b*rho;
    //         pt1.x = std::round(x0 + 1000*(-b));
    //         pt1.y = std::round(y0 + 1000*(a));
    //         pt2.x = std::round(x0 - 1000*(-b));
    //         pt2.y = std::round(y0 - 1000*(a));
    //         cv::line(debuggingRightPos, pt1, pt2, cv::Scalar(128, 255, 128), 2, cv::LINE_AA);
    //         cv::imshow("rpos", debuggingRightPos);
    //     }
    // }

    // PREC y = static_cast<PREC>(mROIHeight) * 0.5f;

    // return std::round((rho - y*std::cos(theta))/std::sin(theta));
}

template <typename PREC>
std::vector<cv::Mat> HoughTransformLaneDetectorThetaparam<PREC>::divideLines(const cv::Mat& image)
{
    cv::Mat objs, stats, centroids;
    int32_t objCnt = cv::connectedComponentsWithStats(image, objs, stats, centroids);

    cv::Mat leftLane = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat rightLane = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat stopLine = cv::Mat::zeros(image.size(), CV_8UC1);

    cv::Mat tuningImage = cv::Mat::zeros(image.size(), CV_8UC3);

    double sumSlY = 0;
    int32_t cntSl = 0;

    if (mDebugging) {
        for (int i = 1; i < 9; i++) {
            cv::line(tuningImage, cv::Point(i*80, 0), cv::Point(i*80, mROIStartHeight-1), cv::Scalar(150, 150, 150));
        }
    }

    for (int i = 1; i < objCnt; i++) {
        int* p = stats.ptr<int>(i);

        double x = centroids.at<double>(i, 0);
        double y = centroids.at<double>(i, 1);

        // objs too small
        if (p[4] < 500 || p[4] > 6000) continue;

        // opting out objects on the floor between lanes
        // **** needs adjusting ****
        // for stop line
        // if (p[0] > 270 && p[0] + p[2] < 370) continue;
        // if(p[1] + p[3] > 25) continue; 
        // std::cout << "p[0] : " << p[0] << std::endl;
        // std::cout << "p[1] : " << p[1] << std::endl;
        // std::cout << "p[2] : " << p[2] << std::endl;
        // std::cout << "p[3] : " << p[3] << std::endl;
        // std::cout << "p[0] + p[2] : " << p[0] + p[2] << std::endl;

        // selecting based on location
        if (p[2] > 200) {
            if (static_cast<int>((leftFirstRange + leftLastRange)/1.5) > static_cast<int>(p[0] + p[2]/2 + 0.5f)){
                cv::Rect bBox(p[0], p[1], p[2], p[3]);
                if (mDebugging) {
                    cv::rectangle(tuningImage, bBox, cv::Scalar(0, 255, 255));
                    cv::putText(tuningImage, std::to_string(p[4]), cv::Point(x,y), 0, 0.4, cv::Scalar(0, 255, 255));
                    //cv::putText(tuningImage, std::to_string(p[3]), cv::Point(x,y), 0, 0.4, cv::Scalar(0, 255, 255));
                }
                cv::Mat tempL = cv::Mat::zeros(image.size(), CV_8UC1);
                image(bBox).copyTo(tempL(bBox));
                std::vector<std::vector<cv::Point>> leftContours;
                std::vector<cv::Vec4i> hierarchyL;
                if (!tempL.empty()) cv::findContours(tempL, leftContours, hierarchyL, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                if (leftContours.empty()) {
                    if (mDebugging) std::cout << "Left lane's contour was not detected" << std::endl;
                    continue;
                }
                cv::drawContours(leftLane, leftContours, -1, cv::Scalar(255), 1);
                cv::rectangle(leftLane, cv::Point(0, 0), cv::Point(mImageWidth, mROIHeight), cv::Scalar(0), 2);
            } else if(static_cast<int>((rightFirstRange + rightLastRange)/1.5) < static_cast<int>(p[0] + p[2]/2 + 0.5f)) {
                cv::Rect bBox(p[0], p[1], p[2], p[3]);
                if (mDebugging) {
                    cv::rectangle(tuningImage, bBox, cv::Scalar(255, 0, 255));
                    cv::putText(tuningImage, std::to_string(p[4]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 0, 255));
                    //cv::putText(tuningImage, std::to_string(p[3]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 0, 255));
                }
                cv::Mat tempR = cv::Mat::zeros(image.size(), CV_8UC1);
                image(bBox).copyTo(tempR(bBox));
                std::vector<std::vector<cv::Point>> rightContours;
                std::vector<cv::Vec4i> hierarchyR;
                if (!tempR.empty()) cv::findContours(tempR, rightContours, hierarchyR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                if (rightContours.empty()) {
                    if (mDebugging) std::cout << "Right lane's contour was not detected" << std::endl;
                    continue;
                }
                cv::drawContours(rightLane, rightContours, -1, cv::Scalar(255), 1);
                cv::rectangle(rightLane, cv::Point(0, 0), cv::Point(mImageWidth, mROIHeight), cv::Scalar(0), 2);
            } else {
                if(p[3] < p3Height && p[4] > minArea && p[4] < maxArea) {
                    cntSl += 1;
                    sumSlY += y;
                    cv::Rect bBox(p[0], p[1], p[2], p[3]);
                    if (mDebugging) {
                        cv::rectangle(tuningImage, bBox, cv::Scalar(255, 255, 0));
                        cv::putText(tuningImage, std::to_string(p[4]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 255, 255));
                    }
                    cv::Mat tempS = cv::Mat::zeros(image.size(), CV_8UC1);
                    // ## DEBUGGING check if one-liner can do ##
                    // cv::Mat bBoxImage = image(bBox).clone();
                    // bBoxImage.copyTo(rightLane(bBox));
                    image(bBox).copyTo(tempS(bBox));
                    std::vector<std::vector<cv::Point>> stopContours;
                    std::vector<cv::Vec4i> hierarchyS;
                    if (!tempS.empty()) cv::findContours(tempS, stopContours, hierarchyS, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    if (stopContours.empty()) {
                        if (mDebugging) std::cout << "Stop line's contour was not detected" << std::endl;
                        continue;
                    }
                    cv::drawContours(stopLine, stopContours, -1, cv::Scalar(255), 1);
                    cv::rectangle(stopLine, cv::Point(0, 0), cv::Point(mImageWidth, mROIHeight), cv::Scalar(0), 2);
                }
            }
        } else {
            if ((static_cast<int>(p[0] + p[2]/2 + 0.5f) < leftFirstRange && p[4] < toDiscard) || (static_cast<int>(p[0] + p[2]/2 + 0.5f) >= leftFirstRange && static_cast<int>(p[0] + p[2]/2 + 0.5f) <= leftLastRange)) {
                cv::Rect bBox(p[0], p[1], p[2], p[3]);
                if (mDebugging) {
                    cv::rectangle(tuningImage, bBox, cv::Scalar(0, 255, 255));
                    cv::putText(tuningImage, std::to_string(p[4]), cv::Point(x,y), 0, 0.4, cv::Scalar(0, 255, 255));
                    //cv::putText(tuningImage, std::to_string(p[3]), cv::Point(x,y), 0, 0.4, cv::Scalar(0, 255, 255));
                }
                cv::Mat tempL = cv::Mat::zeros(image.size(), CV_8UC1);
                image(bBox).copyTo(tempL(bBox));
                std::vector<std::vector<cv::Point>> leftContours;
                std::vector<cv::Vec4i> hierarchyL;
                if (!tempL.empty()) cv::findContours(tempL, leftContours, hierarchyL, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                if (leftContours.empty()) {
                    if (mDebugging) std::cout << "Left lane's contour was not detected" << std::endl;
                    continue;
                }
                cv::drawContours(leftLane, leftContours, -1, cv::Scalar(255), 1);
                cv::rectangle(leftLane, cv::Point(0, 0), cv::Point(mImageWidth, mROIHeight), cv::Scalar(0), 2);
            }
            else if ((static_cast<int>(p[0] + p[2]/2 + 0.5f) > rightLastRange && p[4] < toDiscard) || (static_cast<int>(p[0] + p[2]/2 + 0.5f) <= rightLastRange && static_cast<int>(p[0] + p[2]/2 + 0.5f) >= rightFirstRange)){
                cv::Rect bBox(p[0], p[1], p[2], p[3]);
                if (mDebugging) {
                    cv::rectangle(tuningImage, bBox, cv::Scalar(255, 0, 255));
                    cv::putText(tuningImage, std::to_string(p[4]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 0, 255));
                    //cv::putText(tuningImage, std::to_string(p[3]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 0, 255));
                }
                cv::Mat tempR = cv::Mat::zeros(image.size(), CV_8UC1);
                image(bBox).copyTo(tempR(bBox));
                std::vector<std::vector<cv::Point>> rightContours;
                std::vector<cv::Vec4i> hierarchyR;
                if (!tempR.empty()) cv::findContours(tempR, rightContours, hierarchyR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                if (rightContours.empty()) {
                    if (mDebugging) std::cout << "Right lane's contour was not detected" << std::endl;
                    continue;
                }
                cv::drawContours(rightLane, rightContours, -1, cv::Scalar(255), 1);
                cv::rectangle(rightLane, cv::Point(0, 0), cv::Point(mImageWidth, mROIHeight), cv::Scalar(0), 2);
            }
            else {
                // cv::putText(tuningImage, std::to_string(p[3]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 255, 0));
                if(p[3] < p3Height && p[4] > minArea && p[4] < maxArea) {
                    cntSl += 1;
                    sumSlY += y;
                    cv::Rect bBox(p[0], p[1], p[2], p[3]);
                    if (mDebugging) {
                        cv::rectangle(tuningImage, bBox, cv::Scalar(255, 255, 0));
                        cv::putText(tuningImage, std::to_string(p[4]), cv::Point(x,y), 0, 0.4, cv::Scalar(255, 255, 255));
                    }
                    cv::Mat tempS = cv::Mat::zeros(image.size(), CV_8UC1);
                    // ## DEBUGGING check if one-liner can do ##
                    // cv::Mat bBoxImage = image(bBox).clone();
                    // bBoxImage.copyTo(rightLane(bBox));
                    image(bBox).copyTo(tempS(bBox));
                    std::vector<std::vector<cv::Point>> stopContours;
                    std::vector<cv::Vec4i> hierarchyS;
                    if (!tempS.empty()) cv::findContours(tempS, stopContours, hierarchyS, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    if (stopContours.empty()) {
                        if (mDebugging) std::cout << "Stop line's contour was not detected" << std::endl;
                        continue;
                    }
                    cv::drawContours(stopLine, stopContours, -1, cv::Scalar(255), 1);
                    cv::rectangle(stopLine, cv::Point(0, 0), cv::Point(mImageWidth, mROIHeight), cv::Scalar(0), 2);
                } else continue;
            }
        }
    }

    if (cntSl != 0) mSLpos = static_cast<int>(sumSlY + 0.5f);
    else mSLpos = -1;

    if (mDebugging) tuningImage.copyTo(mDebugObj);

    std::vector<cv::Mat> lanes = {leftLane, rightLane, stopLine};
    return lanes;
    /*
    * dividing left and right lanes based on theta(slope)
    for (uint32_t i = 0; i < linesSize; ++i)
    {
        const auto& line = lines[i];

        PREC rho = line[0];
        PREC theta = line[1];

        // arc-tan(10=mHoughLineSlopeRange) = 1.47112767f
        if (-1.47112767f <= theta && theta < 0.0f)
        {
            //leftLineSumX += static_cast<PREC>(x1 + x2) * 0.5f;
            leftLineIndices.emplace_back(i);
            // DEBUGGING: print rhos and thetas
            std::cout << i << "th line is left lane with theta: " << theta << std::endl;
        }
        else if (0.0f < theta && theta <= 1.47112767f)
        {
            //rightLineSumX += static_cast<PREC>(x1 + x2) * 0.5f;
            rightLineIndices.emplace_back(i);
            // DEBUGGING: print rhos and thetas
            std::cout << i << "th line is right lane with theta: " << theta << std::endl;
        }
    }*/
}

template <typename PREC>
std::pair<int32_t, int32_t> HoughTransformLaneDetectorThetaparam<PREC>::getLanePosition(const cv::Mat& image)
{
    if (mDebugging) image.copyTo(mDebugFrame);

    cv::Mat edgeImage = processRoI(image);

    std::vector<cv::Mat> lanes = divideLines(edgeImage);

    int32_t lpos = getLinePositionX(lanes[0], Direction::LEFT);
    int32_t rpos = getLinePositionX(lanes[1], Direction::RIGHT);


    if ((lpos == 0) || (lpos == 640)) lpos = -1;
    if (rpos == 0 || rpos == 640) rpos = -1;

    // ## DEBUGGING to see if lpos and rpos are right ##
    if (mDebugging) {
        cv::Mat tempL = cv::Mat::zeros(edgeImage.size(), CV_8UC1);
        cv::Mat tempR = cv::Mat::zeros(edgeImage.size(), CV_8UC1);
        
        lanes[0].copyTo(tempL);
        lanes[1].copyTo(tempR);

        cv::Mat debugLeftLane, debugRightLane;
        cv::cvtColor(tempL, debugLeftLane, cv::COLOR_GRAY2BGR);
        cv::cvtColor(tempR, debugRightLane, cv::COLOR_GRAY2BGR);

        cv::Point debugLpos(lpos, static_cast<int>(mROIHeight * 0.5f));
        cv::Point debugRpos(rpos, static_cast<int>(mROIHeight * 0.5f));

        cv::circle(debugLeftLane, debugLpos, 5, cv::Scalar(0, 255, 255),-1);
        cv::circle(debugRightLane, debugRpos, 5, cv::Scalar(255, 0, 255),-1);

        if (!debugLeftLane.empty()) cv::imshow("left", debugLeftLane);
        if (!debugRightLane.empty()) cv::imshow("right", debugRightLane);
    }

    return {lpos, rpos};
}

template <typename PREC>
void HoughTransformLaneDetectorThetaparam<PREC>::drawLines(const Lines& lines, const Indices& leftLineIndices, const Indices& rightLineIndices)
{
    auto draw = [this](const Lines& lines, const Indices& indices) {
        for (const auto index : indices)
        {
            const auto& line = lines[index];
            auto r = static_cast<PREC>(std::rand()) / RAND_MAX * std::numeric_limits<uint8_t>::max();
            auto g = static_cast<PREC>(std::rand()) / RAND_MAX * std::numeric_limits<uint8_t>::max();
            auto b = static_cast<PREC>(std::rand()) / RAND_MAX * std::numeric_limits<uint8_t>::max();

            cv::line(mDebugFrame, { line[static_cast<uint8_t>(HoughIndex::x1)], line[static_cast<uint8_t>(HoughIndex::y1)] + mROIStartHeight },
                     { line[static_cast<uint8_t>(HoughIndex::x2)], line[static_cast<uint8_t>(HoughIndex::y2)] + mROIStartHeight }, { b, g, r }, kDebugLineWidth);
        }
    };

    draw(lines, leftLineIndices);
    draw(lines, rightLineIndices);
}

template <typename PREC>
void HoughTransformLaneDetectorThetaparam<PREC>::drawRectangles(int32_t leftPositionX, int32_t rightPositionX, int32_t estimatedPositionX)
{
    cv::rectangle(mDebugFrame, cv::Point(leftPositionX - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(leftPositionX + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kGreen, kDebugLineWidth);

    cv::rectangle(mDebugFrame, cv::Point(rightPositionX - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(rightPositionX + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kGreen, kDebugLineWidth);

    cv::rectangle(mDebugFrame, cv::Point(estimatedPositionX - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(estimatedPositionX + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kRed, kDebugLineWidth);

    cv::rectangle(mDebugFrame, cv::Point(mImageWidth / 2 - kDebugRectangleHalfWidth, kDebugRectangleStartHeight + mROIStartHeight),
                  cv::Point(mImageWidth / 2 + kDebugRectangleHalfWidth, kDebugRectangleEndHeight + mROIStartHeight), kBlue, kDebugLineWidth);
}

template class HoughTransformLaneDetectorThetaparam<float>;
template class HoughTransformLaneDetectorThetaparam<double>;
} // namespace Xycar
