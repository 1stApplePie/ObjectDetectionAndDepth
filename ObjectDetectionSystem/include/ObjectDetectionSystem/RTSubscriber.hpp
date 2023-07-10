/**
 * @file RTSubscriber.hpp
 * @author Stella Park
 * @author Chanwoo Lee
 * @author Jaejin Kim
 * @brief RTSubscriber class header file
 * @version 0.5
 * @date 2023-07-04
 */

#ifndef RT_SUBSCRIBER_HPP_
#define RT_SUBSCRIBER_HPP_

#include <iostream>
#include <ros/ros.h>

#include <yolov3_trt_ros/BoundingBoxes.h>
#include <yolov3_trt_ros/BoundingBox.h>

namespace Xycar {

/**
 * @brief Tensor RT detection result subscriber
 * @tparam PREC Precision of data
 */
template <typename PREC>
class RTSubscriber final
{
public:
    using Ptr = RTSubscriber*; ///< Pointer type of this class

    /**
     * @brief To cluster discrete Lidar points and find the center of clusters
     *
     * @param[in] config Configuration including parameters for scanner
     */
    RTSubscriber(ros::NodeHandle& nh) { initSub(); }

    /**
     * @brief Get the Debug Frame object pointer
     * 
     * @return Return frame for debuging
     */
    std::vector<std::vector<int32_t>> getbBox();

private:
    /**
     * @brief Set DB as a member variable
     * 
     * @param[in] DB A vector of Lidar Points on a BEV map
     */
    void initSub();

    /**
     * @brief Set DB as a member variable
     * 
     * @param[in] DB A vector of Lidar Points on a BEV map
     */
    void subCallBack(const yolov3_trt_ros::BoundingBoxes &msg);

private:
    // ROS ass.
    ros::NodeHandle nh_;
    ros::Subscriber sub_;

    std::vector<yolov3_trt_ros::BoundingBox> mDataRT;
};
} // namespace Xycar
#endif // RT_SUBSCRIBER_HPP_