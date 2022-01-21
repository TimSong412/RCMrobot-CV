#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <time.h>
using namespace std;

int main()
{
    cv::Mat img, fg, bg, res, finalres;

    img = cv::imread("/home/tim/文档/Pytorch_ws/fasterrcnn_robot/ImgAnno/Anno/1-2/JPGImages/000860.jpg", cv::IMREAD_COLOR);
    // res = cv::Mat(img);[   0  637  212 1854]
    clock_t st, ed;
    double time;
    cv::Rect2d rec(637, 0, 1217, 212);
    st = clock();
    cv::grabCut(img, res, rec, bg, fg, 5, cv::GC_INIT_WITH_RECT);
    ed = clock();
    time = (double)(ed - st) / CLOCKS_PER_SEC;
    cout << time << endl;
    res *= 50;


    cv::imshow("res", res);
    // cout << res << endl;
    cv::waitKey(0);
}
