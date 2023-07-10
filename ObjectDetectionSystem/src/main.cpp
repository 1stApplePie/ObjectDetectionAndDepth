// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file main.cpp
 * @author Jongrok Lee (lrrghdrh@naver.com)
 * @author Chihyeon Lee
 * @brief Lane Keeping System Main Function using Hough Transform
 * @version 1.1
 * @date 2023-05-02
 */
#include "ObjectDetectionSystem/ObjectDetectionSystem.hpp"

using PREC = float;

int32_t main(int32_t argc, char** argv)
{
    ros::init(argc, argv, "Object Detection System");
    Xycar::ObjectDetectionSystem<PREC> objectDetectionSystem;
    objectDetectionSystem.run();

    return 0;
}