#ifndef INC_3DSCENE_COLORLINES_HPP
#define INC_3DSCENE_COLORLINES_HPP

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

// Функция для создания градиента яркости
Mat createGradientRect(int width, int height) {
    Mat gradientRect(height, width, CV_8UC1);

    for (int i = 0; i < gradientRect.cols; i++) {
        for (int j = 0; j < gradientRect.rows; j++) {
            gradientRect.at<uchar>(j, i) = static_cast<uchar>(i * 255 / gradientRect.cols);
        }
    }

    return gradientRect;
}

// Функция для применения гамма-коррекции
Mat applyGammaCorrection(const Mat& input, float gamma) {
    Mat corrected;
    input.convertTo(corrected, CV_32F); // Преобразуем в тип с плавающей точкой
    cv::pow(corrected / 255.0, gamma, corrected);
    corrected *= 255;
    corrected.convertTo(corrected, CV_8U); // Возвращаем к типу uchar
    return corrected;
}

#endif //INC_3DSCENE_COLORLINES_HPP
