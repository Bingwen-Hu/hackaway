#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;


void readPixel(int x, int y, cv::Mat& img){
    cv::Vec3b intensity = img.at<cv::Vec3b> (x, y);
    uchar blue  = intensity[0];
    uchar green = intensity[1];
    uchar red   = intensity[2];

    cout << "Pixel in (" << x << ", " << y << ") is "
         << "(" << (int)blue << ", " << (int)green << ", " << (int)red
         << ")" << endl; 
}

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("/home/mory/Downloads/result/mory.png", -1);
    if ( img.empty() ) return -1;
    
    // cv::namedWindow("cat", cv::WINDOW_AUTOSIZE);
    cv::imshow("cat", img);

    // read pixels
    int x = 16, y = 32;
    readPixel(x, y, img);
    
    // write pixels
    img.at<cv::Vec3b>(x, y) = {0, 1, 2};
    readPixel(x, y, img);


    // wait for user to press a key and then exit
    cv::waitKey(0); 
    cv::destroyWindow("cat");
    return 0;
}