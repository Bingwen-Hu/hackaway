#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("/home/mory/Downloads/cat.jpeg", -1);
    if ( img.empty() ) return -1;
    
    cv::namedWindow("cat", cv::WINDOW_AUTOSIZE);
    cv::imshow("cat", img);
    
    // wait for user to press a key and then exit
    cv::waitKey(0); 
    cv::destroyWindow("cat");
    return 0;
}