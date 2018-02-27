#include <opencv2/opencv.hpp>

void demo( const cv::Mat& image) {
    // create some windows to show the input
    // and output images in.
    cv::namedWindow("Demo-in", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Demo-out", cv::WINDOW_AUTOSIZE);

    // create a window to show our input image
    cv::imshow("Demo-in", image);
    // prepare our output image
    cv::Mat out;

    // Transform, double blur
    cv::GaussianBlur(image, out, cv::Size(5, 5), 3, 3);
    cv::GaussianBlur(  out, out, cv::Size(5, 5), 3, 3);

    // show it!
    cv::imshow("Demo-out", out);

    cv::waitKey(0);
}

int main() {
    cv::Mat img = cv::imread("/home/mory/Downloads/cat.jpeg", -1);
    demo(img);
}