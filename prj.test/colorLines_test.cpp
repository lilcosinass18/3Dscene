#include <colorLines/colorLines.hpp>

int main() {
    int width = 256 * 3;
    int height = 50;

    // Создаем изображение для первого прямоугольника (градиент яркости)
    Mat gradientRect = createGradientRect(width, height);

    // Создаем изображение для второго прямоугольника (с гамма-коррекцией)
    float gamma = 2.2;  // Здесь можно настроить значение гаммы
    Mat gammaCorrectedRect = applyGammaCorrection(gradientRect, gamma);

    // Объединяем оба прямоугольника в одно изображение (по вертикали)
    Mat result(height * 2, width, CV_8UC1);
    gradientRect.copyTo(result(Rect(0, 0, width, height)));
    gammaCorrectedRect.copyTo(result(Rect(0, height, width, height)));

    // Отображаем изображение
    imshow("Two Rectangles", result);
    waitKey(0);

    return 0;
}