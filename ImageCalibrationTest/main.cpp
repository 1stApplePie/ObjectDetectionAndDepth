#include "opencv2/opencv.hpp"
#include <iostream>

constexpr int IMAGE_WIDTH = 640;
constexpr int IMAGE_HEIGHT = 480;

cv::Mat map1, map2;
cv::Rect roi;
cv::Size image_size = cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT);

// cal, dist matrix
double calibrate_mtx_data[9] = {
    374.839208, 0.0, 329.917309,
    0.0, 371.474881, 244.814906,
    0.0, 0.0, 1.0
};

double dist_data[5] = {-0.299279, 0.067634, -0.001693, -0.002189, 0.0};

cv::Mat calibrate_mtx(3, 3, CV_64FC1, calibrate_mtx_data);
cv::Mat dist_coeffs(1, 4, CV_64FC1, dist_data);
cv::Mat camera_matrix = getOptimalNewCameraMatrix(calibrate_mtx, dist_coeffs, image_size, 1, image_size, &roi);

cv::Mat calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2)
{
    // image calibrating
    cv::Mat mapping_image = src.clone();
    cv::Mat calibrated_image;
    remap(src, mapping_image, map1, map2, cv::INTER_LINEAR);
    return calibrated_image;
};

int main()
{
    cv::VideoCapture capture(0);
    cv::initUndistortRectifyMap(calibrate_mtx, dist_coeffs, cv::Mat(), camera_matrix, image_size, CV_32FC1, map1, map2);

    if (!capture.isOpened()){
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    while (true){
        cv::Mat src;
        capture >> src;
        cv::Mat calibrated_image = calibrate_image(src, map1, map2);
        cv::imshow("src", src);
    }
}