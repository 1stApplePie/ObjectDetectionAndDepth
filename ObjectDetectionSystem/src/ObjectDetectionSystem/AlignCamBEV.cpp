/**
 * @file AlignCamBEV.hpp
 * @author Stella Park
 * @author Chanwoo Lee
 * @author Jaejin Kim
 * @brief Align points on BEV with Bbox in cam image
 * @version 0.5
 * @date 2023-07-04
 */

#include <numeric>
#include <map>

#include "ObjectDetectionSystem/AlignCamBEV.hpp"

namespace Xycar {

template <typename PREC>
void AlignCamBEV<PREC>::setVar()
{
    std::vector<float> vec{-1.67589704e-01, -1.15656218e+00,  3.27992827e+02,
        2.10963493e-02, -2.47234187e+00,  6.21059585e+02,
        1.76758126e-05, -4.26389112e-03,  1.00000000e+0};
    cv::Mat(3, 3, CV_32F, vec.data()).copyTo(mH);

    mColorMap = {{0, cv::Scalar(0, 0, 255)}, {1, cv::Scalar(0, 127, 255)}, {2, cv::Scalar(0, 255, 255)}, 
                 {3, cv::Scalar(0, 255, 0)}, {4, cv::Scalar(0, 0, 255)}, {5, cv::Scalar(130, 0, 75)}, 
                 {6, cv::Scalar(211, 0, 148)}}; // left, right, stop, crosswalk, uturn->toy_car, traffic_light, ignore->xy_car, masking->no_color

}

template <typename PREC>
bool AlignCamBEV<PREC>::isInRange(const std::vector<int32_t>& boxDatum, const cv::Point& cCImg)
{
    cv::Point cCenterOnImg = cCImg;
    cv::Rect tmpRect(boxDatum[1], boxDatum[2], boxDatum[3], boxDatum[4]);

    return tmpRect.contains(cCenterOnImg);
}

template <typename PREC>
std::vector<cv::Point> AlignCamBEV<PREC>::transferPT(const std::vector<cv::Point>& cCenters)
{
    std::vector<cv::Point> res;
    for (cv::Point cCenter : cCenters) {
        cv::Vec<float, 3> cBEV(cCenter.x, cCenter.y, 1);
        cv::Mat invH;
        cv::invert(mH, invH);
        cv::Mat cImg = invH*cBEV;
        cImg = cImg / cImg.at<float>(2, 0);

        int tmpX = static_cast<int32_t>(cImg.at<float>(0,0));
        int tmpY = static_cast<int32_t>(cImg.at<float>(1,0));
        cv::Point cCenterImg(tmpX, tmpY);
        res.push_back(cCenterImg);
    }

    return res;
}

template <typename PREC>
std::vector<std::pair<cv::Point, int32_t>> AlignCamBEV<PREC>::assignCls(const std::vector<cv::Point>& cCsImg, const std::vector<std::vector<int32_t>>& boxData)
{
    std::vector<std::pair<cv::Point, int32_t>> res;
    for (cv::Point cCImg : cCsImg) {
        std::vector<int32_t> tmpV;
        for (std::vector<int32_t> boxDatum : boxData) {
            if (isInRange(boxDatum, cCImg)){
                if ((0 <= boxDatum[0]) && (boxDatum[0] < 6)) tmpV = {cCImg.x, cCImg.y, boxDatum[0]};
            }
            else tmpV = {cCImg.x, cCImg.y, 6}; // xy_car
        }
        res.push_back(std::make_pair(cv::Point(tmpV[0], tmpV[1]), tmpV[2]));
    }
    return res;
}

// template <typename PREC>
// cv::Mat AlignCamBEV<PREC>::getBEVmap(const std::vector<cv::Point>& cCenters, const std::vector<std::vector<int32_t>>& boxData, const cv::Mat& img)
// {
//     cv::Mat tmpImg;
//     img.copyTo(tmpImg);

//     // draw lines to make it a grid
//     for (int i = 1; i < 6; i++) {
//         cv::line(tmpImg, cv::Point(i*90, 0), cv::Point(i*90, 539), cv::Scalar(0, 0, 0), 3);
//         cv::line(tmpImg, cv::Point(0, i*90), cv::Point(539, i*90), cv::Scalar(0, 0, 0), 3);
//     }
//     std::vector<cv::Point> tmpC = transferPT(cCenters);
//     std::vector<std::pair<cv::Point, int32_t>> tmp = assignCls(tmpC, boxData);
//     for (std::pair<cv::Point, int32_t> tmpP : tmp) {
//         cv::circle(tmpImg, tmpP.first, 5, mColorMap.at(tmpP.second), -1);
//     }

//     return tmpImg;
// }

template <typename PREC>
cv::Mat AlignCamBEV<PREC>::getBEVmap(const std::vector<cv::Point>& cCenters, const std::vector<std::vector<int32_t>>& boxData, const cv::Mat& debugImg)
{
    cv::Mat tmpImg;
    debugImg.copyTo(tmpImg);

    for (cv::Point cC : cCenters) {
        cv::circle(tmpImg, cC, 3, cv::Scalar(255, 255, 0), -1);
    }

    for (std::vector<int32_t> datum : boxData) {
        cv::rectangle(tmpImg, cv::Rect(datum[1], datum[2], datum[3], datum[4]), cv::Scalar(255, 0, 255), 2);
    }

    return tmpImg;
}

template <typename PREC>
std::vector<std::pair<cv::Point, int32_t>> AlignCamBEV<PREC>::getBEVpts(const std::vector<cv::Point>& cCenters, const std::vector<std::vector<int32_t>>& boxData)
{
    std::vector<cv::Point> tmpC = transferPT(cCenters);
    std::vector<std::pair<cv::Point, int32_t>> res = assignCls(tmpC, boxData);

    return res;
}

template class AlignCamBEV<float>;
template class AlignCamBEV<double>;
} // namespace Xycar