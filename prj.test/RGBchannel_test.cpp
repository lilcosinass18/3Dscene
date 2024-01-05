#include <RGBchannel/RGBchannel.hpp>

int main() {
    // Загрузка исходной цветной фотографии
    Mat originalImage = imread("../images/monkey.jpg");

    if (originalImage.empty()) {
        std::cerr << "Ошибка загрузки фотографии." << std::endl;
        return -1;
    }

    // Создание коллажа из фотографии и каналов цветов
    Mat collage = createColorChannelCollage(originalImage);

    // Отображение и сохранение коллажа
    imshow("Collage", collage);


    waitKey(0);

    return 0;
}
