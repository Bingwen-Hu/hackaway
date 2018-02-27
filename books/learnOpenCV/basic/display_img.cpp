#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("/home/mory/Downloads/cat.jpeg", -1);
    if ( img.empty() ) return -1;
    
    cv::namedWindow("show image", cv::WINDOW_AUTOSIZE);
    cv::imshow("cat", img);
    cv::waitKey(0);
    cv::destroyWindow("show image");
    return 0;
}