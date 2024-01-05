#ifndef INC_3DSCENE_RGBCHANNEL_HPP
#define INC_3DSCENE_RGBCHANNEL_HPP

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;


Mat createColorChannelCollage(const Mat& inputImage) {

    // Размер уменьшенной фотографии
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Создание изображений для красного, зеленого и синего каналов
    Mat redChannel(height, width, CV_8UC3, Scalar(0, 0, 0));
    Mat greenChannel(height, width, CV_8UC3, Scalar(0, 0, 0));
    Mat blueChannel(height, width, CV_8UC3, Scalar(0, 0, 0));

    // Копирование соответствующих каналов из исходной фотографии
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Vec3b pixel = inputImage.at<Vec3b>(i, j);
            redChannel.at<Vec3b>(i, j) = Vec3b(pixel[0], 0, 0);       // Красный канал
            greenChannel.at<Vec3b>(i, j) = Vec3b(0, pixel[1], 0);     // Зеленый канал
            blueChannel.at<Vec3b>(i, j) = Vec3b(0, 0, pixel[2]);      // Синий канал
        }
    }

    // Создание коллажа из четырех изображений
    Mat collage(height * 2, width * 2, CV_8UC3);
    inputImage.copyTo(collage(Rect(0, 0, width, height)));
    redChannel.copyTo(collage(Rect(width, 0, width, height)));
    greenChannel.copyTo(collage(Rect(0, height, width, height)));
    blueChannel.copyTo(collage(Rect(width, height, width, height)));

    return collage;
}

#endif //INC_3DSCENE_RGBCHANNEL_HPP
