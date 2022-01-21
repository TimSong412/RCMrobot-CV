#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <time.h>
using namespace std;

int main()
{
    cv::Mat img1, img2, descriptors1, descriptors2, final;
    float k_data[3][3] = {{486.9320, 0, 342.5736}, {0, 489.3566, 193.4255}, {0, 0, 1}};
    cv::Mat K = cv::Mat(3, 3, CV_32F, k_data);

    img1 = cv::imread("../imgpairs/img4.jpg", cv::IMREAD_COLOR);
    img2 = cv::imread("../imgpairs/img5.jpg", cv::IMREAD_COLOR);
    // img1 = cv::imread("/home/tim/文档/Pytorch_ws/robotcv_pipeline/region5.png", cv::IMREAD_COLOR);
    // img2 = cv::imread("/home/tim/文档/Pytorch_ws/robotcv_pipeline/region6.png", cv::IMREAD_COLOR);

    clock_t st, ed;
    double time;
    st = clock();
    vector<cv::KeyPoint> kp1, kp2;
    // cv::Ptr<cv::ORB> detector = cv::ORB::create();
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    detector->detectAndCompute(img1, cv::Mat(), kp1, descriptors1);
    detector->detectAndCompute(img2, cv::Mat(), kp2, descriptors2);
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);

    vector<vector<cv::DMatch>> matches;
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5), cv::makePtr<cv::flann::SearchParams>(32));
    
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    vector<cv::DMatch> final_matches;
    double maxdist = -1, rate = 0.7;
    std::vector<cv::Point2d> pts1, pts2;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance / matches[i][1].distance < rate)
        {
            final_matches.emplace_back(matches[i][0]);
            pts1.emplace_back(kp1[matches[i][0].queryIdx].pt);
            pts2.emplace_back(kp2[matches[i][0].trainIdx].pt);
        }
    }
    const int pts_size = (int)pts1.size();

    cv::Mat inliers;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::FM_RANSAC, 0.99, 1.0, inliers);
    // cout << inliers << endl;
    cv::Mat R1, R2, t, R0, t0;
    cv::decomposeEssentialMat(E, R1, R2, t);
    cv::recoverPose(E, pts1, pts2, K, R0, t0);
    ed = clock();
    time = (double)(ed - st) / CLOCKS_PER_SEC;
    cout << "time= " << time << endl;

    cv::drawMatches(img1, kp1, img2, kp2, final_matches, final);
    cv::imshow("final", final);
    cv::waitKey(0);

    cv::FileStorage fs("transform.xml", cv::FileStorage::WRITE);
    fs << "R1" << R1 << "R2" << R2 << "t" << t << "R0" << R0 << "t0" << t0;
    fs.release();
}