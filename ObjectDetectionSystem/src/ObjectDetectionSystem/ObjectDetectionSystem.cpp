// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file ObjectDetectionSystem.cpp
 * @author Jongrok Lee (lrrghdrh@naver.com)
 * @author Jiho Han
 * @author Haeryong Lim
 * @author Chihyeon Lee
 * @brief Lane Keeping System Class source file
 * @version 1.1
 * @date 2023-05-02
 */
#include <cmath>
#include "ObjectDetectionSystem/ObjectDetectionSystem.hpp"

namespace Xycar {
template <typename PREC>
ObjectDetectionSystem<PREC>::ObjectDetectionSystem()
{
    std::string configPath;
    mNodeHandler.getParam("config_path", configPath);
    YAML::Node config = YAML::LoadFile(configPath);

    mPID = new PIDController<PREC>(config["PID"]["P_GAIN"].as<PREC>(), config["PID"]["I_GAIN"].as<PREC>(), config["PID"]["D_GAIN"].as<PREC>());
    mMovingAverage = new MovingAverageFilter<PREC>(config["MOVING_AVERAGE_FILTER"]["SAMPLE_SIZE"].as<uint32_t>());
    mHoughTransformLaneDetector = new HoughTransformLaneDetectorThetaparam<PREC>(config);
    mClusterLidarPoints = new ClusterLidarPoints<PREC>(config);
    mRTSubscriber = new RTSubscriber<PREC>(mNodeHandler);
    mAlignCamBEV = new AlignCamBEV<PREC>();
    setParams(config);

    mPublisher = mNodeHandler.advertise<xycar_msgs::xycar_motor>(mPublishingTopicName, mQueueSize);
    mSubscriber = mNodeHandler.subscribe(mSubscribedTopicName, mQueueSize, &ObjectDetectionSystem::imageCallback, this);
    mScanscriber = mNodeHandler.subscribe("/scan", 1000, &ObjectDetectionSystem::scanCallback, this);
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::setParams(const YAML::Node& config)
{
    mPublishingTopicName = config["TOPIC"]["PUB_NAME"].as<std::string>();
    mSubscribedTopicName = config["TOPIC"]["SUB_NAME"].as<std::string>();
    mQueueSize = config["TOPIC"]["QUEUE_SIZE"].as<uint32_t>();
    mXycarSpeed = config["XYCAR"]["START_SPEED"].as<PREC>();
    mXycarMaxSpeed = config["XYCAR"]["MAX_SPEED"].as<PREC>();
    mXycarMinSpeed = config["XYCAR"]["MIN_SPEED"].as<PREC>();
    mXycarSpeedControlThreshold = config["XYCAR"]["SPEED_CONTROL_THRESHOLD"].as<PREC>();
    mAccelerationStep = config["XYCAR"]["ACCELERATION_STEP"].as<PREC>();
    mDecelerationStep = config["XYCAR"]["DECELERATION_STEP"].as<PREC>();
    mCteParams = config["CTE"]["CTE_ERROR"].as<PREC>();
    mDebugging = config["DEBUG"].as<bool>();
    //mDebugStopping = config["DEBUG"]["STOP"].as<bool>();
    //mDebugToyCar = config["DEBUG"]["TOYCAR"].as<bool>();
    // DEBUGGING STOP FOR THE SIGNS

    // DEBUGGING DETOUR THE TOY CARS

    // DEBUGGING DYNAMIC OBSTACLES
    // mDynamicAngle = config["DYNAMIC"]["ANGLE"].as<uint32_t>();
    // mDynamicTh = config["DYNAMIC"]["NUM_OBSTACLES"].as<uint32_t>();
    // mDynamicMInRange = config["DYNAMIC"]["MIN_RANGE"].as<PREC>();
    // mDynamicMaxRange = config["DYNAMIC"]["MAX_RANGE"].as<PREC>();
}

template <typename PREC>
ObjectDetectionSystem<PREC>::~ObjectDetectionSystem()
{
    delete mPID;
    delete mMovingAverage;
    delete mHoughTransformLaneDetector;
    delete mClusterLidarPoints;
    delete mRTSubscriber;
    delete mAlignCamBEV;
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::run()
{
    double calibrate_mtx_data[9] = {
        374.839208, 0.0, 329.917309,
        0.0, 371.474881, 244.814906,
        0.0, 0.0, 1.0
    };

    double dist_data[5] = { -0.299279, 0.067634, -0.001693, -0.002189, 0.0};

    cv::Rect roi;
    cv::Mat map1, map2;
    cv::Mat calibrate_mtx(3, 3, CV_64FC1, calibrate_mtx_data);
    cv::Mat distCoeffs(1, 4, CV_64FC1, dist_data);
    cv::Size image_size = cv::Size(640, 480);
    cv::Mat cameraMatrix = getOptimalNewCameraMatrix(calibrate_mtx, distCoeffs, image_size, 1, image_size, &roi);

    cv::initUndistortRectifyMap(calibrate_mtx, distCoeffs, cv::Mat(), cameraMatrix, image_size, CV_32FC1, map1, map2);

    ros::Rate rate(kFrameRate);
    
    mLaneSpace = 450;

    while (ros::ok())
    {     
        ros::spinOnce();
        if (mFrame.empty())
            continue;
        // cv::imshow("frame", mFrame);

        // HANDLING DYNAMIC OBSTACLES
        // if (handleDynamicObs(mLidarData)) {
        //     if (mDebugging) std::cout << "STOPPING FOR DYNAMIC OBSTACLE" << std::endl;
        //     mPublisher.publish(mMotorMessage);
        //     continue;
        // }

        calibrate_image(mFrame, map1, map2, roi).copyTo(mCalibratedFrame);
        cv::Mat warped_image = warp_image(mCalibratedFrame);

        const auto [pointed_img, pointBEV] = drawLidarPoint(warped_image, mLidarData);
        if (mDebugging) cv::imshow("pointed", pointed_img);
        
        auto [leftPositionX, rightPositionX] = mHoughTransformLaneDetector->getLanePosition(mCalibratedFrame);

        if (mDebugging) {
            std::cout << "########DEBUGGING BEFORE POSX CORRECTION IS DONE" << std::endl;
            std::cout << "lpos: " << leftPositionX << ", rpos: " << rightPositionX 
                      << ", mpos: " << static_cast<int32_t>((leftPositionX + rightPositionX)/2 + 0.5f) 
                      << std::endl;
        }

        mMpos = correctMissingLanes(leftPositionX, rightPositionX);
        if (mDebugging) {
            std::cout << "########DEBUGGING AFTER POSX CORRECTION IS DONE" << std::endl;
            std::cout << "lpos: " << leftPositionX << ", rpos: " << rightPositionX << ", mpos: " << mMpos << std::endl;
        }

        if (!pointBEV.empty()) mCCenters = mClusterLidarPoints->getCenters(pointBEV);
        else mCCenters = std::vector<cv::Point>();

	    mBBox = mRTSubscriber->getbBox();
        if (mBBox.empty()) {
            // Without Detection Result
            driveN(mMpos);
            if (mDebugging) {
                std::cout << "No objects are being detected" << std::endl;
                std::cout << "DEBUGGING DRIVING MODE: NORMAL WITH SPEED: " << mMotorMessage.speed << ", ANGLE: " << mMotorMessage.angle << std::endl;
            }
            mPublisher.publish(mMotorMessage);
        } 
        else {
            // With Objects Detected
            int32_t mStatus = 0;

            // ## DEBUGGING
            // cv::Mat debugTmp = mAlignCamBEV->getBEVmap(mCCenters, mBBox, pointBEV);
            // cv::imshow("BEV",debugTmp);
            // cv::Mat debugTmp = mAlignCamBEV->getBEVmap(mCCenters, mBBox, mCalibratedFrame);
            // cv::imshow("IMG",debugTmp);
            // cv::waitKey(10);

            std::vector<std::pair<cv::Point, int32_t>> objBEV = mAlignCamBEV->getBEVpts(mCCenters, mBBox);
            std::vector<float> objD = getDistances(objBEV);

            // int32_t min_idx = min_element(objD.begin(), objD.end()) - objD.begin();
            // std::cout << "nearest box is " << min_idx + 1 << std::endl;
            // int32_t clsID = objBEV[min_idx].second;

            // ##DEBUGGING FILTERING WITH BOX WIDTH-->NOT OPTIMAL NEEDS ADJUSTING
            std::vector<int32_t> tmpW;
            for (std::vector<int32_t> boxT : mBBox) {
                tmpW.push_back(boxT[3]);
            }
            int32_t maxIdx = max_element(tmpW.begin(), tmpW.end()) - tmpW.begin();

            std::vector<int32_t> objoI= mBBox[maxIdx];
            if (mDebugging) {
                std::cout << "nearest box is " << maxIdx + 1 << std::endl;
                std::cout << "class id for nearest box is " << objoI[0] << std::endl;
            }
            driveO(objoI);
        }

        if (mDebugging)
        {
            std::cout << "########DEBUGGING AFTER ALL THE DRIVING DECISIONS ARE MADE" << std::endl;
            std::cout << "lpos: " << mLpos << ", rpos: " << mRpos << ", mpos: " << mMpos << std::endl;
            // mHoughTransformLaneDetector->drawRectangles(mLpos, mRpos, mMpos);
            
            // to show calibrated raw image
            cv::imshow("Debug", mHoughTransformLaneDetector->getDebugFrame());
            // to show binarization of RoI
            cv::imshow("RoI", mHoughTransformLaneDetector->getDebugRoI()); 
            // to fine tune selection standard for lanes and stop-line
            cv::imshow("Obj", mHoughTransformLaneDetector->getDebugObj());
            cv::waitKey(1);
            std::cout << std::endl;
        }
    }
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::imageCallback(const sensor_msgs::Image& message)
{
    cv::Mat src = cv::Mat(message.height, message.width, CV_8UC3, const_cast<uint8_t*>(&message.data[0]), message.step);
    cv::cvtColor(src, mFrame, cv::COLOR_RGB2BGR);
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::scanCallback(const sensor_msgs::LaserScan::ConstPtr& lidar_message)
{
    mLidarData = lidar_message->ranges;
}


template <typename PREC>
cv::Mat ObjectDetectionSystem<PREC>::calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2, cv::Rect const& roi)
{
	// image calibrating
    cv::Size image_size = cv::Size(640, 480);
	cv::Mat mapping_image = src.clone();
	cv::Mat calibrated_image;
	remap(src, mapping_image, map1, map2, cv::INTER_LINEAR);

	mapping_image = mapping_image(roi);
	resize(mapping_image, calibrated_image, image_size);
	return calibrated_image;
}

template <typename PREC>
cv::Mat ObjectDetectionSystem<PREC>::warp_image(cv::Mat image)
{
    int warp_image_width = 540;
	int warp_image_height = 540;

    float homography_data[9] = {
        -1.67589704e-01, -1.15656218e+00,  3.27992827e+02,
        2.10963493e-02, -2.47234187e+00,  6.21059585e+02,
        1.76758126e-05, -4.26389112e-03,  1.00000000e+0
    };

    cv::Mat homography_matrix(3, 3, CV_32FC1, homography_data);

	cv::Mat warped_image;
	warpPerspective(image, warped_image, homography_matrix, cv::Size(warp_image_width, warp_image_height), cv::INTER_CUBIC);

	return warped_image;
}
template <typename PREC>
std::pair<cv::Mat,std::vector<cv::Point>> ObjectDetectionSystem<PREC>::drawLidarPoint(const cv::Mat& image, std::vector<float>& lidar_data)
{
    int data_size = lidar_data.size();
    auto start = lidar_data.begin();
    auto end = lidar_data.begin() + data_size;

    int warp_image_width = 540;
    int warp_image_height = 540;

    std::vector<cv::Point> pointBEV;
    // pointBEV.resize(169);

    if (data_size==0) return {image, {cv::Point(-1, -1)}};

    else {
        std::vector<float> front_left_data = std::vector<float>(lidar_data.begin(), lidar_data.begin() + data_size/6 + 1);
        std::vector<float> front_right_data = std::vector<float>(lidar_data.begin() + data_size*5/6+1, lidar_data.end());

        for(int i = data_size/6 + 1; i > 0; i--)
        {
            float range = front_left_data.back()*200;
            front_left_data.pop_back();

            if (range == 0.f) continue;

            double sint, cost;
            if (i == 0) 
            {
                sint = 0.0;
                cost = 0.0;
            }
            else
            {
                double theta = 2*pi/data_size*i;
                sint = std::sin(-theta);
                cost = std::cos(-theta);
            }

            cv::Point center = cv::Point(static_cast<int32_t>(warp_image_width/2 + range*sint),
                                         static_cast<int32_t>(warp_image_height - range*cost));
            pointBEV.push_back(center);
            cv::circle(image, center, 1, cv::Scalar(0, 0, 255));
        }

        for(int j = 0; j < data_size/6; j++)
        {
            float range = front_right_data.back()*200;
            front_right_data.pop_back();

            if (range == 0.f) continue;
            
            double sint, cost;
            if (j == 0) 
            {
                sint = 0.0;
                cost = 0.0;
            }
            else
            {
                double theta = 2*pi/data_size*j;
                sint = std::sin(theta);
                cost = std::cos(theta);
            }

            cv::Point center = cv::Point(static_cast<int32_t>(warp_image_width/2 + range*sint),
                                         static_cast<int32_t>(warp_image_height - range*cost));
            pointBEV.push_back(center);
            cv::circle(image, center, 1, cv::Scalar(0, 255, 0));
        }

        return {image, pointBEV};
    }
}

template <typename PREC>
int32_t ObjectDetectionSystem<PREC>::correctMissingLanes(int32_t lpos, int32_t rpos)
{
    int32_t tmpMpos;
    if ((lpos != -1) || (rpos != -1)) 
    {
        if (lpos != -1 && rpos != -1) {
            // mLaneSpace = static_cast<int32_t>(rpos - lpos + 0.5f);
            tmpMpos = (static_cast<int32_t>((lpos + rpos) / 2));
        }
        else if (lpos == -1) 
            lpos = rpos - mLaneSpace;
        else if (rpos == -1) 
            rpos = lpos + mLaneSpace;
        tmpMpos = static_cast<int32_t>((lpos + rpos) / 2);
    } else tmpMpos = static_cast<int32_t>(mCalibratedFrame.cols / 2);

    mLpos = lpos;
    mRpos = rpos;

    return tmpMpos;
}

template <typename PREC>
std::vector<float> ObjectDetectionSystem<PREC>::getDistances(std::vector<std::pair<cv::Point, int32_t>>& objs)
{
    std::vector<float> res;
    for (std::pair<cv::Point, int32_t> obj : objs) {
        // assume xy_car point is at (270, 540)
        res.push_back(std::sqrt(std::pow(obj.first.x - 270, 2) + std::pow(obj.first.y - 540, 2)));
    }
    return res;
}

template <typename PREC>
bool ObjectDetectionSystem<PREC>::handleDynamicObs(std::vector<float>& lidar_data)
{
    int32_t data_size = lidar_data.size();
    // int32_t chunk = static_cast<int32_t>(mDynamicAngle / 720 + 0.5f);

    std::vector<float> front_left_data = std::vector<float>(lidar_data.begin(), lidar_data.begin() + data_size/6 + 1);
    std::vector<float> front_right_data = std::vector<float>(lidar_data.begin() + data_size*5/6+1, lidar_data.end());

    int32_t cntObs = 0;

    for(int i = data_size/6 + 1; i > 0; i--) {
        float range = front_left_data.back()*200;
        front_left_data.pop_back();

        if ((range == 0.f) || (i == 0)) continue;
        if ((0.05 <= range) && (range <= 0.3)) cntObs += 1;
    }

    for(int j = 0; j < data_size/6; j++) {
        float range = front_right_data.back()*200;
        front_right_data.pop_back();

        if (range == 0.f) continue;
        if ((0.05 <= range) && (range <= 0.3)) cntObs += 1;
    }
    
    if (cntObs > 6) {
        mMotorMessage.angle = 0;
        mMotorMessage.speed = 0;
        return true;
    }
    return false;
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::handleRoundabout(int32_t direction, const std::vector<int32_t>& obj)
{
    if (mDebugging) std::cout << "Handling Roundabout, FOLLOWING SINGLE LANE TO STEER" << std::endl;
    if (direction == -1) {
        if (((mLpos == -1) && (mRpos == -1)) && (obj[3] > 44)) {
            driveN(0);
            mPublisher.publish(mMotorMessage);
            ros::Duration(0.3).sleep();
            driveN(0);
        } else mMpos = mLpos + 100;
        if (mDebugging) {
            std::cout << "LEFT ROUNDING" << std::endl;
            std::cout << "width: " << obj[3] <<std::endl;
        }
    }
    else if (direction == 1) {
        if (((mLpos == -1) && (mRpos == -1)) && (obj[3] > 44)){
            driveN(640);
            mPublisher.publish(mMotorMessage);
            ros::Duration(1.0).sleep();
            driveN(640);
        } else mMpos = mRpos - 100;
        if (mDebugging) {
            std::cout << "RIGHT ROUNDING" << std::endl;
            std::cout << "width: " << obj[3] <<std::endl;
        } 
    }

    return;
}

template <typename PREC>
bool ObjectDetectionSystem<PREC>::handleHault()
{
    if (mDebugging) std::cout << "Hault Sign is Processed" << std::endl;
    if (mHoughTransformLaneDetector->getStopLineY() == -1)
    {
        // Stop Line is not visible yet in RoI
        driveN(mMpos);
        return false;
    } else {
        return true;
    }
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::handleToyCar()
{
    // where to face
    int32_t orientX = static_cast<int32_t>(mCalibratedFrame.cols / 2);

    if (mDebugging) std::cout << "Detouring the toy cars" << std::endl;

    std::vector<std::vector<int32_t>> toyCars;
    for (std::vector<int32_t> tmpBox : mBBox) {
        if (tmpBox[0] == 4) toyCars.push_back(tmpBox);
    }

    int32_t numCars = toyCars.size();

    if (numCars > 1) {
        // 2 toy cars are visible
        int32_t hindCarIdx;
        if (toyCars[0][3] < toyCars[1][3]) hindCarIdx = 0;
        else hindCarIdx = 1;
        orientX = static_cast<int32_t>(toyCars[hindCarIdx][1] + toyCars[hindCarIdx][3]/2 + 0.5f) - 120;
    } else if (numCars == 1) { // 1, 0
        // only 1 toy car is visible
        if (toyCars[0][3] < 120) orientX = static_cast<int32_t>(toyCars[0][1] + toyCars[0][3]/2 + 0.5f);
    }

    mMpos = orientX;
    int32_t errorFromMid = mMpos - static_cast<int32_t>(mCalibratedFrame.cols / 2);
    errorFromMid = static_cast<PREC>(mPID->getControlOutput(errorFromMid)); 
    speedControl(errorFromMid);

    mMotorMessage.angle = std::round(errorFromMid);
    mMotorMessage.speed = std::round(mXycarSpeed);
}


template <typename PREC>
bool ObjectDetectionSystem<PREC>::handleTrafficLight(const std::vector<int32_t>& tlBox)
{
    cv::Mat roiTL;
    mCalibratedFrame(cv::Rect(tlBox[1], tlBox[2], tlBox[3], tlBox[4])).copyTo(roiTL);
    
    //##DEBUGGING
    if (mDebugging) {
        std::cout << "Traffic light is Processed" << std::endl;
        cv::imshow("handle", roiTL);
	    std::cout << "at objD file x and y min: " << tlBox[1] << ", " << tlBox[2] << std::endl;
        std::cout << "at objD file w and h: " << tlBox[3] << ", " << tlBox[3] << std::endl;
    }
	
    cv::Mat hls;
    cv::cvtColor(roiTL, hls, cv::COLOR_BGR2HLS);

    cv::Mat tmpRes;
    cv::inRange(hls, cv::Scalar(0, 200, 200), cv::Scalar(25, 255, 255), tmpRes);

    int32_t cntR = cv::countNonZero(tmpRes);

    //##DEBUGGING
    if (mDebugging) cv::imshow("handleR", tmpRes);
    
    if (cntR > 5 && tlBox[3] > 80) return true; // red light
    else {                     // green or yellow light
        driveN(mMpos);
        return false; 
    }     
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::speedControl(PREC steeringAngle)
{
    if (std::abs(steeringAngle) > mXycarSpeedControlThreshold)
    {
        mXycarSpeed -= mDecelerationStep;
        mXycarSpeed = std::max(mXycarSpeed, mXycarMinSpeed);
	    mXycarSpeed = std::min(mXycarSpeed, mXycarMaxSpeed);
        return;
    }
    else {
        mXycarSpeed += mAccelerationStep;
        mXycarSpeed = std::min(mXycarSpeed, mXycarMaxSpeed);
	    return;
    }
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::driveN(int32_t mposEstimation)
{
    int32_t errorFromMid = mposEstimation - static_cast<int32_t>(mCalibratedFrame.cols / 2);
    errorFromMid = static_cast<PREC>(mPID->getControlOutput(errorFromMid));
    speedControl(errorFromMid);
    mMotorMessage.angle = std::round(errorFromMid);
    mMotorMessage.speed = std::round(mXycarSpeed);     
}

template <typename PREC>
void ObjectDetectionSystem<PREC>::driveO(std::vector<int32_t> nearObj)
{
    // Setting Xycar MSG according to the detected Traffic Sign
    if (nearObj[0] == 0)
    {
        handleRoundabout(-1, nearObj);
    }
    else if (nearObj[0] == 1)
    {
        handleRoundabout(1, nearObj);
    }
    else if (nearObj[0] == 2)
    {
        bool is_stop = handleHault();
        if (is_stop) {
            if (mDebugging) std::cout << "stopping for signs 2" << std::endl; 
            mMotorMessage.angle = 0;
            mMotorMessage.speed = 0;
            mPublisher.publish(mMotorMessage);
            ros::Duration(5.0).sleep();
            driveN(mMpos);
            mPublisher.publish(mMotorMessage);
            ros::Duration(1.0).sleep();
            driveN(mMpos);
        }
    }
    else if (nearObj[0] == 3)
    {
        bool is_stop = handleHault();
        if (is_stop) {
            if (mDebugging) std::cout << "stopping for signs 3" << std::endl; 
            mMotorMessage.angle = 0;
            mMotorMessage.speed = 0;
            mPublisher.publish(mMotorMessage);
            ros::Duration(5.0).sleep();
            driveN(mMpos);
            mPublisher.publish(mMotorMessage);
            ros::Duration(1.0).sleep();
            driveN(mMpos);
        }
    }
    else if (nearObj[0] == 4)
    {
        //handleToyCar();
        if (mDebugging) std::cout << "TOY CAR DETECTED" << std::endl;
        driveN(mMpos);
    }
    else if (nearObj[0] == 5)
    {
        bool is_red = handleTrafficLight(nearObj);
        if (is_red) {
            if (mDebugging) std::cout << "stopping for red light" << std::endl;
            mMotorMessage.angle = 0;
            mMotorMessage.speed = 0;
        }
    }
    else
    {
        if (mDebugging) std::cout << "NOSIGNIFICANT OBJ IS DETECTED, NORMAL DRIVING MODE IS ON" << std::endl;
        driveN(mMpos);
    }

    if (mDebugging) std::cout << "DEBUGGING DRIVING MODE: OBJECT WITH SPEED: " << mMotorMessage.speed << ", ANGLE: " << mMotorMessage.angle << std::endl;
    mPublisher.publish(mMotorMessage);
}

template class ObjectDetectionSystem<float>;
template class ObjectDetectionSystem<double>;
} // namespace Xycar