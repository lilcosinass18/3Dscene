#ifndef INC_3DSCENE_NOISE_HPP
#define INC_3DSCENE_NOISE_HPP

#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

cv::Mat createGradientSquaresImage(int img_size, int square_size) {
    // Создаем черно-белое изображение
    cv::Mat image = cv::Mat::zeros(cv::Size(img_size, img_size), CV_8UC1);

    // Проценты белого для каждого квадрата
    int percentages[4][4] = {
            {0, 5, 15, 20},
            {25, 30, 35, 40},
            {45, 50, 55, 60},
            {65, 75, 80, 85}
    };

    // Заполняем квадраты соответствующими оттенками серого
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            // Преобразуем процент в значение яркости
            uchar brightness = static_cast<uchar>((percentages[i][j] / 100.0) * 255);
            // Заполняем квадрат
            cv::rectangle(image,
                          cv::Point(j * square_size, i * square_size),
                          cv::Point((j + 1) * square_size, (i + 1) * square_size),
                          cv::Scalar(brightness),
                          cv::FILLED);
        }
    }

    // Отображаем изображение
    cv::imwrite("orig_image.jpg", image);

    return image;
}

cv::Mat addSaltAndPepperNoise(const cv::Mat& image, double saltProb, double pepperProb) {
    cv::Mat noisyImage = image.clone();
    cv::RNG rng;

    int numPixels = image.rows * image.cols;

    for (int i = 0; i < numPixels; ++i) {
        double randomValue = rng.uniform(0.0, 1.0);

        if (randomValue < saltProb) {
            noisyImage.at<uchar>(i / image.cols, i % image.cols) = 255;  // Salt noise
        } else if (randomValue > (1.0 - pepperProb)) {
            noisyImage.at<uchar>(i / image.cols, i % image.cols) = 0;    // Pepper noise
        }
    }

    imwrite("salt&paper_image.jpg", noisyImage);
    return noisyImage;
}

cv::Mat addSinusoidalNoise(const cv::Mat& image, double amplitude, double frequency) {
    cv::Mat noisyImage = image.clone();

    for (int i = 0; i < noisyImage.rows; ++i) {
        for (int j = 0; j < noisyImage.cols; ++j) {
            double noise = amplitude * sin(2.0 * CV_PI * frequency * i / noisyImage.rows);
            noisyImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(noisyImage.at<uchar>(i, j) + noise);
        }
    }


    imwrite("sinusoidalImage.jpg", noisyImage);
    return noisyImage;
}

cv::Mat addMultiplicativeNoise(const cv::Mat& image, double scale) {
    cv::Mat noisyImage = image.clone();
    cv::RNG rng;

    for (int i = 0; i < noisyImage.rows; ++i) {
        for (int j = 0; j < noisyImage.cols; ++j) {
            double randomValue = rng.uniform(0.0, 1.0);
            double noise = -log(randomValue) * scale;

            // Умножение яркости пикселя на шум
            noisyImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(noisyImage.at<uchar>(i, j) * noise);
        }
    }

    imwrite("multiplicativeImage.jpg", noisyImage);
    return noisyImage;
}

cv::Mat addGaussianNoise(const cv::Mat& image, double mean, double stddev) {
    cv::Mat noisyImage = image.clone();
    cv::RNG rng;
    cv::Mat noise(noisyImage.size(), CV_8UC1);
    rng.fill(noise, cv::RNG::NORMAL, mean, stddev);

    cv::addWeighted(noisyImage, 1.0, noise, 1.0, 0.0, noisyImage);
    imwrite("gaussianImage.jpg", noisyImage);
    return noisyImage;
}

// Функция для применения фильтра среднего значения
void meanFilter(const cv::Mat& input, cv::Mat& output) {
    cv::blur(input, output, cv::Size(3, 3)); // использование ядра 3x3
}

// Функция для применения медианного фильтра
void medianFilter(const cv::Mat& input, cv::Mat& output) {
    cv::medianBlur(input, output, 3); // размер ядра 3
}

// Функция для применения билатерального фильтра
void bilateralFilter(const cv::Mat& input, cv::Mat& output) {
    cv::bilateralFilter(input, output, 9, 75, 75); // диаметр = 9, sigmaColor = 75, sigmaSpace = 75
}


#endif //INC_3DSCENE_NOISE_HPP
