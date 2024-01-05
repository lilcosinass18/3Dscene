#include <opencv2/opencv.hpp>
#include <iostream>

double calculatePSNR_HVS_M(const cv::Mat& original, const cv::Mat& modified) {
    if (original.size() != modified.size() || original.type() != modified.type()) {
        throw std::invalid_argument("Input images must have the same size and type");
    }

    // Конвертируем изображения в формат с плавающей точкой (32 бита на канал)
    cv::Mat originalFloat, modifiedFloat;
    original.convertTo(originalFloat, CV_32F);
    modified.convertTo(modifiedFloat, CV_32F);

    // Рассчитываем PSNR-HVS-M
    cv::Mat diff = originalFloat - modifiedFloat;
    cv::Mat mseMap = diff.mul(diff);

    // Коэффициенты моделирования визуальной системы человека
    const double c1 = 3.25 * 3.25;
    const double c2 = 0.0173 * 0.0173;
    const double c3 = 1.4096;
    const double c4 = 1.0 / (0.002 * 255 * 0.002 * 255);

    // Рассчитываем метрику для каждого канала
    cv::Scalar mseScalar = cv::mean(mseMap);

    double mse = mseScalar[0] + mseScalar[1] + mseScalar[2];
    double psnr_hvs_m = 10.0 * log10((255 * 255) / (mse + c1 * c2));

    // Коррекция на основе моделирования визуальной системы человека
    psnr_hvs_m += c3 * exp(-c4 * psnr_hvs_m);

    return psnr_hvs_m;
}

// Функция для создания гистограммы для одного канала и вычисления CDF
std::pair<cv::Mat, cv::Mat> createHistogramAndCDF(const cv::Mat& channel, const cv::Scalar& color) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist, cdf;

    cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    hist.copyTo(cdf);

    for (int i = 1; i < histSize; i++) {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }

    cdf /= cdf.at<float>(histSize - 1);

    int hist_w = 256, hist_h = 256;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < histSize; i++) {
        cv::rectangle(histImage,
                      cv::Point(bin_w * i, hist_h),
                      cv::Point(bin_w * i + bin_w - 1, hist_h - cvRound(hist.at<float>(i))),
                      color, -1, 8, 0);
    }
    return {histImage, cdf};
}

// Функция для нахождения пороговых значений alpha1 и alpha2 на основе CDF
void findAlphaThresholds(const cv::Mat& cdf, double& alpha1, double& alpha2, double lower_percent, double upper_percent) {
    alpha1 = lower_percent * 255;
    alpha2 = upper_percent * 255;

    for (int i = 0; i < 256; i++) {
        if (cdf.at<float>(i) >= lower_percent) {
            alpha1 = i;
            break;
        }
    }

    for (int i = 255; i >= 0; i--) {
        if (cdf.at<float>(i) <= upper_percent) {
            alpha2 = i;
            break;
        }
    }
}

// Функция для применения алгоритма CLAHE к изображению
cv::Mat applyCLAHE(const cv::Mat& inputImage, double clipLimit = 2.0, cv::Size gridSize = cv::Size(8, 8)) {
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_planes;
    cv::split(labImage, lab_planes);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, gridSize);
    clahe->apply(lab_planes[0], lab_planes[0]);

    cv::merge(lab_planes, labImage);

    cv::Mat claheImage;
    cv::cvtColor(labImage, claheImage, cv::COLOR_Lab2BGR);

    return claheImage;
}

// Функция для создания изображения с гистограммами и CDF
cv::Mat createCombinedImage(const cv::Mat& srcImage, const cv::Mat& b_hist, const cv::Mat& g_hist, const cv::Mat& r_hist,
                            const cv::Mat& b_cdf, const cv::Mat& g_cdf, const cv::Mat& r_cdf) {
    cv::Mat combinedImage(512, 512, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Size newSize(256, 256);

    cv::Mat resizedImage;
    cv::resize(srcImage, resizedImage, newSize);
    cv::Mat topLeftROI = combinedImage(cv::Rect(0, 0, 256, 256));
    resizedImage.copyTo(topLeftROI);

    cv::Mat topRightROI = combinedImage(cv::Rect(256, 0, 256, 256));
    r_hist.copyTo(topRightROI);

    cv::Mat bottomLeftROI = combinedImage(cv::Rect(0, 256, 256, 256));
    g_hist.copyTo(bottomLeftROI);

    cv::Mat bottomRightROI = combinedImage(cv::Rect(256, 256, 256, 256));
    b_hist.copyTo(bottomRightROI);

    return combinedImage;
}

int main() {
    cv::Mat srcImage = cv::imread("../images/lemonade.jpg");
    if (srcImage.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> bgr_planes;
    cv::split(srcImage, bgr_planes);

    cv::Mat b_hist, g_hist, r_hist, b_cdf, g_cdf, r_cdf;
    std::tie(b_hist, b_cdf) = createHistogramAndCDF(bgr_planes[0], cv::Scalar(255, 0, 0));
    std::tie(g_hist, g_cdf) = createHistogramAndCDF(bgr_planes[1], cv::Scalar(0, 255, 0));
    std::tie(r_hist, r_cdf) = createHistogramAndCDF(bgr_planes[2], cv::Scalar(0, 0, 255));

    cv::Mat combinedImage = createCombinedImage(srcImage, b_hist, g_hist, r_hist, b_cdf, g_cdf, r_cdf);

    double alpha1_b, alpha2_b, alpha1_g, alpha2_g, alpha1_r, alpha2_r;
    findAlphaThresholds(b_cdf, alpha1_b, alpha2_b, 0.07, 0.93);
    findAlphaThresholds(g_cdf, alpha1_g, alpha2_g, 0.07, 0.93);
    findAlphaThresholds(r_cdf, alpha1_r, alpha2_r, 0.07, 0.93);

    std::cout << "Blue channel: alpha1 = " << alpha1_b << ", alpha2 = " << alpha2_b << std::endl;
    std::cout << "Green channel: alpha1 = " << alpha1_g << ", alpha2 = " << alpha2_g << std::endl;
    std::cout << "Red channel: alpha1 = " << alpha1_r << ", alpha2 = " << alpha2_r << std::endl;

    cv::Mat contrasted_b, contrasted_g, contrasted_r;
    bgr_planes[0].convertTo(contrasted_b, CV_8UC1, 255.0 / (alpha2_b - alpha1_b), -alpha1_b * 255.0 / (alpha2_b - alpha1_b));
    bgr_planes[1].convertTo(contrasted_g, CV_8UC1, 255.0 / (alpha2_g - alpha1_g), -alpha1_g * 255.0 / (alpha2_g - alpha1_g));
    bgr_planes[2].convertTo(contrasted_r, CV_8UC1, 255.0 / (alpha2_r - alpha1_r), -alpha1_r * 255.0 / (alpha2_r - alpha1_r));

    cv::Mat contrasted_image;
    std::vector<cv::Mat> contrasted_planes = {contrasted_b, contrasted_g, contrasted_r};
    cv::merge(contrasted_planes, contrasted_image);

    cv::Mat b_hist_contrast, g_hist_contrast, r_hist_contrast,
            b_cdf_contrast, g_cdf_contrast, r_cdf_contrast;
    std::tie(b_hist_contrast, b_cdf_contrast) = createHistogramAndCDF(contrasted_b, cv::Scalar(255, 0, 0));
    std::tie(g_hist_contrast, g_cdf_contrast) = createHistogramAndCDF(contrasted_g, cv::Scalar(0, 255, 0));
    std::tie(r_hist_contrast, r_cdf_contrast) = createHistogramAndCDF(contrasted_r, cv::Scalar(0, 0, 255));

    cv::Mat combinedImageContrasted = createCombinedImage(contrasted_image, b_hist_contrast, g_hist_contrast,
                                                          r_hist_contrast, b_cdf_contrast, g_cdf_contrast, r_cdf_contrast);

    cv::Mat clahe_image = applyCLAHE(srcImage);

    std::vector<cv::Mat> bgr_planes_clahe;
    cv::split(clahe_image, bgr_planes_clahe);

    cv::Mat b_hist_clahe, g_hist_clahe, r_hist_clahe, b_cdf_clahe, g_cdf_clahe, r_cdf_clahe;
    std::tie(b_hist_clahe, b_cdf_clahe) = createHistogramAndCDF(bgr_planes_clahe[0], cv::Scalar(255, 0, 0));
    std::tie(g_hist_clahe, g_cdf_clahe) = createHistogramAndCDF(bgr_planes_clahe[1], cv::Scalar(0, 255, 0));
    std::tie(r_hist_clahe, r_cdf_clahe) = createHistogramAndCDF(bgr_planes_clahe[2], cv::Scalar(0, 0, 255));

    cv::Mat combinedImageClahe = createCombinedImage(clahe_image, b_hist_clahe, g_hist_clahe, r_hist_clahe, b_cdf_clahe, g_cdf_clahe, r_cdf_clahe);

    std::cout << "PSNR_HVS_M for autocontrast: " << calculatePSNR_HVS_M(srcImage, contrasted_image) << std::endl;
    std::cout << "PSNR_HVS_M for CLAHE: " << calculatePSNR_HVS_M(srcImage, clahe_image) << std::endl;

    cv::imwrite("../transformed_images/combined_image_with_histograms.png", combinedImage);
    cv::imwrite("../transformed_images/combined_contrasted_image_with_histograms.png", combinedImageContrasted);
    cv::imwrite("../transformed_images/combined_imageCLAHE_with_histograms.png", combinedImageClahe);

    cv::imwrite("../transformed_images/autcontrasted_image.png", contrasted_image);
    cv::imwrite("../transformed_images/clahe_image.png", clahe_image);

    cv::imshow("Original Image", srcImage);
    cv::imshow("Autocorrected Image", contrasted_image);
    cv::imshow("CLAHE Image", clahe_image);

    int key = cv::waitKey(0);
    if (key == 27) {  // Код клавиши Esc
        cv::destroyAllWindows();
        return 0;
    }

    return 0;
}


