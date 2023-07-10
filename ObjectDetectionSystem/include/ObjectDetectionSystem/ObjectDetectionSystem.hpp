#ifndef OBJECT_DETECTION_SYSTEM_HPP_
#define OBJECT_DETECTION_SYSTEM_HPP_

#include <algorithm>
#include <ros/ros.h>
#include <cmath>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <xycar_msgs/xycar_motor.h>
#include <yaml-cpp/yaml.h>

#include "ObjectDetectionSystem/AlignCamBEV.hpp"
#include "ObjectDetectionSystem/RTSubscriber.hpp"
#include "ObjectDetectionSystem/ClusterLidarPoints.hpp"
#include "ObjectDetectionSystem/HoughTransformLaneDetectorThetaparam.hpp"
#include "ObjectDetectionSystem/MovingAverageFilter.hpp"
#include "ObjectDetectionSystem/PIDController.hpp"

namespace Xycar {

/**
 * @brief Lane Keeping System for searching and keeping Hough lines using Hough, Moving average and PID control
 *
 * @tparam Precision of data
 */
template <typename PREC>
class ObjectDetectionSystem
{
public:
    using Ptr = ObjectDetectionSystem*;                                     ///< Pointer type of this class
    using ControllerPtr = typename PIDController<PREC>::Ptr;            ///< Pointer type of PIDController
    using FilterPtr = typename MovingAverageFilter<PREC>::Ptr;          ///< Pointer type of MovingAverageFilter
    using DetectorPtr = typename HoughTransformLaneDetectorThetaparam<PREC>::Ptr; ///< Pointer type of LaneDetector
    using ClustererPtr = typename ClusterLidarPoints<PREC>::Ptr;
    using SubscriberPtr = typename RTSubscriber<PREC>::Ptr;
    using AlignerPtr = typename AlignCamBEV<PREC>::Ptr;

    static constexpr int32_t kXycarSteeringAangleLimit = 50; ///< Xycar Steering Angle Limit
    static constexpr double kFrameRate = 33.0;               ///< Frame rate
    static constexpr double pi = 3.141592;
    /**
     * @brief Construct a new Lane Keeping System object
     */
    ObjectDetectionSystem();

    /**
     * @brief Destroy the Lane Keeping System object
     */
    virtual ~ObjectDetectionSystem();

    /**
     * @brief Run Lane Keeping System
     */
    void run();

private:
    /**
     * @brief Set the parameters from config file
     *
     * @param[in] config Configuration for searching and keeping Hough lines using Hough, Moving average and PID control
     */
    void setParams(const YAML::Node& config);



    /**
     * @brief Callback function for image topic
     *
     * @param[in] message Image topic message
     */
    void imageCallback(const sensor_msgs::Image& message);
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& lidar_message);

    cv::Mat calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2, cv::Rect const& roi);
    cv::Mat warp_image(cv::Mat image);

    std::pair<cv::Mat,std::vector<cv::Point>> drawLidarPoint(const cv::Mat& image, std::vector<float>& lidar_data);

    /**
     * @brief Initialize posX for missing lanes
     *
     * @param[in] lpos, rpos Hough detected posX, -1 if missing
     * @return calculated mpos
     */
    int32_t correctMissingLanes(int32_t lpos, int32_t rpos);
    std::vector<float> getDistances(std::vector<std::pair<cv::Point, int32_t>>& objs);

    /**
     * @brief Handle situations encounterring various objects
     */
    bool handleDynamicObs(std::vector<float>& lidar_data); ///< true if Obstacle is present
    void handleRoundabout(int32_t direction, const std::vector<int32_t>& obj); ///< set mpos accordingly
    bool handleHault();                         ///< true when the stop line is visible
    void handleToyCar();                        ///< set mpos so the toy cars are detoured
    bool handleTrafficLight(const std::vector<int32_t>& tlBox);                  ///< true if Red Light

    /**
     * @brief Control the speed of xycar
     *
     * @param[in] steeringAngle Angle to steer xycar. If over max angle, deaccelerate, otherwise accelerate
     */
    void speedControl(PREC steeringAngle);

    /**
     * @brief Publish the motor topic message
     *
     * @param[in] nearObj ClsID for the detected object
     */
    void driveO(std::vector<int32_t> nearObj);

    /**
     * @brief Set the motor topic message
     *
     * @param[in] mposEstimation CTE is calculated according to param
     */
    void driveN(int32_t mposEstimation);

private:
    // member Pointers
    ControllerPtr mPID;                      ///< PID Class for Control
    FilterPtr mMovingAverage;                ///< Moving Average Filter Class for Noise filtering
    DetectorPtr mHoughTransformLaneDetector; ///< Hough Transform Lane Detector Class for Lane Detection
    ClustererPtr mClusterLidarPoints;
    SubscriberPtr mRTSubscriber;
    AlignerPtr mAlignCamBEV;

    // ROS Variables
    ros::NodeHandle mNodeHandler;          ///< Node Hanlder for ROS. In this case Detector and Controler
    ros::Publisher mPublisher;             ///< Publisher to send message about
    ros::Subscriber mSubscriber;           ///< Subscriber to receive image
    ros::Subscriber mScanscriber;          ///< Subscriber to receive LiDAR Info
    std::string mPublishingTopicName;      ///< Topic name to publish
    std::string mSubscribedTopicName;      ///< Topic name to subscribe
    uint32_t mQueueSize;                   ///< Max queue size for message
    xycar_msgs::xycar_motor mMotorMessage; ///< Message for the motor of xycar

    // OpenCV Image processing Variables
    cv::Mat mFrame; ///< Image from camera. The raw image is converted into cv::Mat
    cv::Mat mCalibratedFrame;

    // Xycar Device variables
    PREC mXycarSpeed;                 ///< Current speed of xycar
    PREC mXycarMaxSpeed;              ///< Max speed of xycar
    PREC mXycarMinSpeed;              ///< Min speed of xycar
    PREC mXycarSpeedControlThreshold; ///< Threshold of angular of xycar
    PREC mAccelerationStep;           ///< How much would accelrate xycar depending on threshold
    PREC mDecelerationStep;           ///< How much would deaccelrate xycar depending on threshold
    PREC mCteParams; 

    // LiDAR variables
    std::vector<float> mLidarData;

    // Debug Flag
    bool mDebugging;     ///< to Debug or not
    //bool mDebugStopping; ///< Debug stopping for stop/ crosswalk/ traffic light
    //bool mDebugToyCar;   ///< Debug detour mechanism for the toy cars 

    // Variables used for motor msg setting
    int32_t mLpos;
    int32_t mRpos;
    int32_t mMpos;

    int32_t mLaneSpace;
    std::vector<cv::Point> mCCenters;
    std::vector<std::vector<int32_t>> mBBox;

    // Variables for stopping for the signs

    // Variables for detouring the toy cars

    // Variables for handling dynamic obstacles
    // uint32_t mDynamicAngle;
    // uint32_t mDynamicTh;
    // PREC mDynamicMInRange;
    // PREC mDynamicMaxRange; 
};
} // namespace Xycar
# endif