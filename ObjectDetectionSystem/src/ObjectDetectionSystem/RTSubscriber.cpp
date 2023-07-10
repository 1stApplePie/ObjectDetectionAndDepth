/**
 * @file RTSubscriber.hpp
 * @author Stella Park
 * @author Chanwoo Lee
 * @author Jaejin Kim
 * @brief RTSubscriber class header file
 * @version 0.5
 * @date 2023-07-04
 */

#include <numeric>

#include "ObjectDetectionSystem/RTSubscriber.hpp"

namespace Xycar {

template <typename PREC>
std::vector<std::vector<int32_t>> RTSubscriber<PREC>::getbBox()
{
    // [clsID, x, y, w, h]
    std::vector<std::vector<int32_t>> res;

    if (mDataRT.empty()) {
        std::vector<int32_t> tmp(5, -1);
        res.push_back(tmp);
        return res;
    }
    else 
    {
        for (size_t i=0; i < mDataRT.size(); i++) {
            int32_t tmpX = mDataRT[i].xmin;
            tmpX = std::max(tmpX, 0);
            int32_t tmpY = mDataRT[i].ymin;
            tmpY = std::max(tmpY, 0);
        
            int32_t tmpW = mDataRT[i].xmax - mDataRT[i].xmin;
            tmpW = std::max(tmpW, 0);
            int32_t tmpH = mDataRT[i].ymax - mDataRT[i].ymin;
            tmpH = std::max(tmpH, 0);

            std::cout << "at sub file x and y min: " << mDataRT[i].xmin << ", " << mDataRT[i].ymin << std::endl;
            std::vector<int32_t> tmpResV = {mDataRT[i].id, tmpX, tmpY, tmpW, tmpH};
            res.push_back(tmpResV);
        }
        return res;
    }
}

template <typename PREC>
void RTSubscriber<PREC>::initSub()
{
    sub_ = nh_.subscribe("/yolov3_trt_ros/detections", 1, &RTSubscriber::subCallBack, this);
}

template <typename PREC>
void RTSubscriber<PREC>::subCallBack(const yolov3_trt_ros::BoundingBoxes &msg)
{
    mDataRT = msg.bounding_boxes;
    // std::cout << "callback fn called by subscriber" << std::endl;
}

template class RTSubscriber<float>;
template class RTSubscriber<double>;
} // namespace Xycar
