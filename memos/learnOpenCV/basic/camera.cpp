/**
 * very important thing: there should be a camera!
 */

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    cv::namedWindow("Demo", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;

    if (argc == 1){
        cap.open(0);
    } else {
        cap.open(argv[1]);
    }

    if ( !cap.isOpened() ) {
        std::cerr << "Couldn't open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    for (;;) {
        cap >> frame;
        if ( frame.empty() ) break;
        cv::imshow("video", frame);

        // wait 33ms, if user press a key then exit
        // if not start the loop again
        if (cv::waitKey(33) >= 0) break;
    }
}