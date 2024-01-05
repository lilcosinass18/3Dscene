#include <noise/noise.hpp>

int main() {

    Mat originalImage = createGradientSquaresImage(400,100);

    // Параметры соль и перец шума
    double saltProbability = 0.01;
    double pepperProbability = 0.01;

    // Добавление соль и перец шума
    cv::Mat saltAndPaperNoisyImage = addSaltAndPepperNoise(originalImage, saltProbability, pepperProbability);

    // Амплитуда и частота синусоидального шума
    double noiseAmplitude = 40.0;
    double noiseFrequency = 0.9;

    // Добавление синусоидального шума
    cv::Mat sinusoidalImage = addSinusoidalNoise(originalImage, noiseAmplitude, noiseFrequency);


    // Параметр масштабирования шума
    double scale = 0.1;

    // Добавление умножительного шума
    cv::Mat MultiplicativeNoisyImage = addMultiplicativeNoise(originalImage, scale);

    // Параметры гауссовского шума
    double mean = 0.0;
    double stddev = 25.0;

    // Добавление гауссовского шума
    cv::Mat GaussianNoisyImage = addGaussianNoise(originalImage, mean, stddev);


    cv::Mat resultMean;
    cv::Mat resultMedian;
    cv::Mat resultBilateral;

    //salt and paper
    meanFilter(saltAndPaperNoisyImage, resultMean);
    medianFilter(saltAndPaperNoisyImage, resultMedian);
    bilateralFilter(saltAndPaperNoisyImage, resultBilateral);

    cv::imwrite("meanFilteredSaltAndPaper.png", resultMean);
    cv::imwrite("medianFilteredSaltAndPaper.png", resultMedian);
    cv::imwrite("bilateralFilteredSaltAndPaper.png", resultBilateral);

    meanFilter(MultiplicativeNoisyImage, resultMean);
    medianFilter(MultiplicativeNoisyImage, resultMedian);
    bilateralFilter(MultiplicativeNoisyImage, resultBilateral);

    cv::imwrite("meanFilteredMult.png", resultMean);
    cv::imwrite("medianFilteredMult.png", resultMedian);
    cv::imwrite("bilateralFilteredMult.png", resultBilateral);


    return 0;
}