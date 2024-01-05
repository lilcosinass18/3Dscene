//#include <colorHistogram/colorHistogram.hpp>
//
//int main() {
//    // Load the image
//    cv::Mat src = cv::imread("../images/monkey.jpg");
//
//    if (src.empty()) {
//        std::cerr << "Image not found." << std::endl;
//        return -1;
//    }
//
//    const size_t number_of_channels = src.channels();
//    const cv::Scalar background_colour(0, 0, 0);
//
//    std::vector<cv::Mat> split;
//    cv::split(src, split);
//
//    const int height = 480;
//    const int width = 480;
//    const int histogram_size = 256;
//    const float range[] = {0, 256};
//    const float* ranges = {range};
//    const bool uniform = true;
//    const bool accumulate = false;
//    cv::Mat mask;
//
//    const int margin = 3;
//    const int min_y = margin;
//    const int max_y = height - margin;
//    const int thickness = 1;
//    const int line_type = cv::LINE_AA;
//    const float bin_width = static_cast<float>(width) / static_cast<float>(histogram_size);
//    cv::Mat dst(height + src.rows, width, CV_8UC3, background_colour); // Create the output image with extra space for the image
//
//    cv::Scalar colours[] = {
//            {255, 0, 0},  // Blue
//            {0, 255, 0},  // Green
//            {0, 0, 255}   // Red
//    };
//
//    if (number_of_channels == 1) {
//        // For grayscale images, use black or white for the histogram
//        colours[0] = (background_colour == cv::Scalar(0, 0, 0)) ?
//                     cv::Scalar(255, 255, 255) :
//                     cv::Scalar(0, 0, 0);
//    }
//
//    // Copy the original image to the top of the destination image
//    src.copyTo(dst(cv::Rect(0, 0, src.cols, src.rows)));
//
//    // Iterate through all the channels in this image
//    for (size_t idx = 0; idx < split.size(); idx++) {
//        const cv::Scalar colour = colours[idx % 3];
//
//        cv::Mat& m = split[idx];
//
//        cv::Mat histogram;
//        cv::calcHist(&m, 1, 0, mask, histogram, 1, &histogram_size, &ranges, uniform, accumulate);
//
//        cv::normalize(histogram, histogram, 0, dst.rows - src.rows, cv::NORM_MINMAX);
//
//        for (int i = 1; i < histogram_size; i++) {
//            const int x1 = std::round(bin_width * (i - 1));
//            const int x2 = std::round(bin_width * i);
//
//            const int y1 = std::clamp(height - static_cast<int>(std::round(histogram.at<float>(i - 1))), min_y, max_y);
//            const int y2 = std::clamp(height - static_cast<int>(std::round(histogram.at<float>(i))), min_y, max_y);
//
//            cv::line(dst, cv::Point(x1, y1 + src.rows), cv::Point(x2, y2 + src.rows), colour, thickness, line_type);
//        }
//    }
//
//    // Display the result
//    cv::imshow("Image with Histograms", dst);
//    cv::waitKey(0);
//
//    return 0;
//}
//
//
//
//
//
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <math.h>

// namespace
using namespace std;
using namespace cv;

cv::Mat equalizeHist(cv::Mat &img);
cv::Mat calculateHistogramImage(const cv::Mat &src);


int main()
{

    Mat img, hist_img;

    // loaded the image
    img = imread("../images/mad_wheels.png"); // FOR WINDOWS USERS "IMAGE_GRAYSCALE"

    if(img.empty()) return -1;

    // Input results
    cv::imshow("Before histogram equalization", img);
    cv::waitKey();

    hist_img = calculateHistogramImage(img);
    imshow("Histogram before equalization", hist_img);
    waitKey(0);


    // Output results
    cv::Mat img_out, out_img_hist;
    img_out = equalizeHist(img);

    imshow("After histogram equalization", img_out);
    waitKey(0);

    out_img_hist = calculateHistogramImage(img_out);
    imshow("Histogram after equalization", out_img_hist);
    waitKey(0);

}


cv::Mat equalizeHist(cv::Mat &img){
    // Total number of occurance of the number of each pixels at different levels from 0 - 256
    // Flattening our 2d matrix
    int flat_img[256] = {0};
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            int index;
            index = static_cast<int>(img.at<uchar>(i,j)); // safe convertion to int
            flat_img[index]++;
        }
    }

    // calculate CDF corresponding to flat_img
    // CDF --> cumsum
    int cumsum[256]={0};
    int memory=0;
    for(int i=0; i<256; i++){
        memory += flat_img[i];
        cumsum[i] = memory;
    }

    // using general histogram equalization formula
    int normalize_img[256]={0};
    for(int i=0; i<256; i++){
        // norm(v) = round(((cdf(v) - mincdf) / (M * N) - mincdf) * (L - 1));
        normalize_img[i] = ((cumsum[i]-cumsum[0])*255)/(img.rows*img.cols-cumsum[0]);
        normalize_img[i] = static_cast<int>(normalize_img[i]);
    }

    // convert 1d back into a 2d matrix
    cv::Mat result(img.rows, img.cols, CV_8U);

    Mat_<uchar>::iterator itr_result = result.begin<uchar>(); // our result
    Mat_<uchar>::iterator it_begin = img.begin<uchar>(); // beginning of the image
    Mat_<uchar>::iterator itr_end = img.end<uchar>(); // end of the image

    for(; it_begin!=itr_end; it_begin++){
        int intensity_value = static_cast<int>(*it_begin); // get the value and cast it into an int
        *itr_result = normalize_img[intensity_value];
        itr_result++;
    }


    return result;
}

cv::Mat calculateHistogramImage(const cv::Mat &img){
    // calculate histogram
    int histSize = 256; // size of the histogram
    float range[] = {0, 255}; //begin and end
    const float* histRange = {range};
    Mat hist_img;
    calcHist(&img, 1, 0, Mat(), hist_img, 1, &histSize, &histRange);

    // draw
    cv::Mat dst(256, 256, CV_8UC1, Scalar(255));
    float max = 0;
    for(int i=0; i<histSize; i++){
        if( max < hist_img.at<float>(i))
            max = hist_img.at<float>(i);
    }

    float scale = (0.9*256)/max;
    for(int i=0; i<histSize; i++){
        int intensity = static_cast<int>(hist_img.at<float>(i)*scale);
        line(dst,Point(i,255),Point(i,255-intensity),Scalar(0));
    }
    return dst;

}
