/**
 * GUI event loop is tricky, just for a taste 
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;

int g_slider_position = 0;
int g_run = 1;
int g_dontset = 0; // start out in single step mode
cv::VideoCapture g_cap;

void onTrackbarSlide(int pos, void *) {
    g_cap.set(cv::CAP_PROP_POS_FRAMES, pos);
    if ( !g_dontset ){
        g_run = 1;
    }
    g_dontset = 0;
}


int main(int argc, char** argv){
    cv::namedWindow("Demos", cv::WINDOW_AUTOSIZE);
    g_cap.open(argv[1]);
    
    int frames_count = (int) g_cap.get(cv::CAP_PROP_FRAME_COUNT);
    int tmp_width    = (int) g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int tmp_height   = (int) g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cout << "Video has "  << frames_count << " frames of dimensions( "
         << tmp_width << ", " << tmp_height  << ")." << endl;
    cv::createTrackbar("Position", "Demos", &g_slider_position,
                       frames_count, onTrackbarSlide);

    cv::Mat frame;
    for (;;) {
        if ( g_run != 0) {
            g_cap >> frame;
            if ( frame.empty() ) break;

            int current_pos = (int) g_cap.get(cv::CAP_PROP_POS_FRAMES);
            g_dontset = 1;
            cv::setTrackbarPos("Position", "Demos", current_pos);
            cv::imshow("Demos", frame);

            g_run -= 1;
        }

        char c = (char) cv::waitKey(10);
        if ( c == 's' ) { // single step
            g_run = 1; 
            cout << "Single step, run = " << g_run << endl;
        }
        if ( c == 'r' ) {
            g_run = -1; 
            cout << "Run mode, run = " << g_run << endl;
        }
        if ( c == 27 ) {
            break;
        }
    }
    return 0;
}