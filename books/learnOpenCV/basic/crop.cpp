#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    auto img = cv::imread("/home/mory/Downloads/result/mory.png", -1);

    auto crop = img(cv::Rect(10, 20, 30, 40));
    cv::imshow("org", img);
    cv::imshow("crop", crop);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}