#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

int main(int argc, char** argv){
    // namedWindow("video", WINDOW_AUTOSIZE);
    VideoCapture cap;
    cap.open(argv[1]);

    Mat frame;
    for (;;) {
        cap >> frame;
        if ( frame.empty() ) break;
        imshow("video", frame);

        // wait 33ms, if user press a key then exit
        // if not start the loop again
        if (waitKey(33) >= 0) break;
    }

    return 0;
}