#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    auto img = cv::imread("/home/mory/Downloads/result/mory.png", -1);
    int x = 10, y = 20, w = 30, h = 40;
    auto crop = img(cv::Rect(x, y, w, h));
    cv::imshow("org", img);
    cv::imshow("crop", crop);

    auto crop_ = img(cv::Rect(y, x, h, w));
    cv::imshow("crop_", crop_);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}