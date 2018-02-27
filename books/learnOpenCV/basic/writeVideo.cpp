#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    cv::namedWindow("Org", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("New", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap(argv[1]); // open video

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size size(
        (int) cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    cv::VideoWriter writer;
    writer.open(argv[2], CV_FOURCC('M', 'J', 'P', 'G'), fps, size);

    cv::Mat logpolar_frame, bgr_frame;
    for (;;) {
        cap >> bgr_frame;
        if ( bgr_frame.empty() ) break; // end if done

        cv::imshow("Org", bgr_frame);
        cv::logPolar(
            bgr_frame,
            logpolar_frame,
            cv::Point2f(bgr_frame.cols/2, bgr_frame.rows/2),
            40,
            cv::WARP_FILL_OUTLIERS
        );

        cv::imshow("New", logpolar_frame);
        writer << logpolar_frame;

        char c = cv::waitKey(10);
        if ( c == 27 ) break;
    }

    cap.release();
}