#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    auto img = cv::imread("/home/mory/Downloads/result/mory.png", -1);
    cv::Mat img180, img90, imgNeg90;
    cv::flip(img, img180, 0);
    cv::transpose(img, img90);
    cv::flip(img90, imgNeg90, 0);

    cv::imshow("org", img);
    cv::imshow("180", img180);
    cv::imshow("90", img90);
    cv::imshow("Neg90", imgNeg90);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}