#ifndef INC_3DSCENE_COLORHISTOGRAM_HPP
#define INC_3DSCENE_COLORHISTOGRAM_HPP

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

// Функция для создания изображения с гистограммами
Mat createHistogramImage(const Mat& inputImage) {
    // Разделение исходной фотографии на каналы RGB
    std::vector<Mat> channels;
    split(inputImage, channels);

    // Вычисление гистограмм для каждого канала
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;

    Mat redHist, greenHist, blueHist;
    calcHist(&channels[0], 1, 0, Mat(), redHist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[1], 1, 0, Mat(), greenHist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[2], 1, 0, Mat(), blueHist, 1, &histSize, &histRange, uniform, accumulate);

    // Нормализация гистограмм до диапазона [0, 1]
    int histHeight = 256;
    normalize(redHist, redHist, 0, histHeight, NORM_MINMAX, -1, Mat());
    normalize(greenHist, greenHist, 0, histHeight, NORM_MINMAX, -1, Mat());
    normalize(blueHist, blueHist, 0, histHeight, NORM_MINMAX, -1, Mat());

    // Создание изображения для гистограмм
    int histWidth = inputImage.cols;
    Mat histogramImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));

    // Отрисовка гистограмм на изображении
    for (int i = 1; i < histSize; i++) {
        line(histogramImage, Point(i - 1, histHeight - cvRound(redHist.at<float>(i - 1))),
             Point(i, histHeight - cvRound(redHist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
        line(histogramImage, Point(i - 1, histHeight - cvRound(greenHist.at<float>(i - 1))),
             Point(i, histHeight - cvRound(greenHist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
        line(histogramImage, Point(i - 1, histHeight - cvRound(blueHist.at<float>(i - 1))),
             Point(i, histHeight - cvRound(blueHist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
    }

    return histogramImage;
}


#endif //INC_3DSCENE_COLORHISTOGRAM_HPP
