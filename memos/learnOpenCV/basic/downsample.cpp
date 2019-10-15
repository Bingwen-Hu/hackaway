#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img1, img2;

    cv::namedWindow("Image1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Image2", cv::WINDOW_AUTOSIZE);

    img1 = cv::imread("/home/mory/Downloads/result/mory.png");
    cv::imshow("Image1", img1);
    cv::pyrDown(img1, img2);
    cv::imshow("Image2", img2);

    cv::waitKey(0);
    return 0;
}