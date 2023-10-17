#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
using namespace cv;
void load_image_and_preprocess()
{
    // load image
    Mat origin_img = imread("hat.jpg");
    int image_width = 640;
    int image_height = 384;
    int image_channel = 3;
    int image_origin_width=origin_img.size().width;
    int image_origin_height=origin_img.size().height;
    // bgr to rgb
    cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2RGB);
    cv::Mat resized_img;
    float scale_my = fmin(image_width*1.0/image_origin_width,image_height*1.0/image_origin_height);
    int new_image_width = scale_my*image_origin_width;
    int new_image_height = scale_my*image_origin_height;
    cv::resize(origin_img, resized_img, cv::Size(new_image_width,new_image_height));
    cv::Mat image_padded;
    image_padded = cv::Mat(image_height, image_width, CV_32FC3, cv::Scalar(128.0, 128.0, 128.0));
    int dw = (image_width - new_image_width) / 2;
    int dh = (image_height - new_image_height) / 2;
    cv::Rect roi(dw, dh, new_image_width, new_image_height);
    resized_img.copyTo(image_padded(roi));
    image_padded.convertTo(image_padded, CV_32FC3);
    image_padded /= 255.0;
    cv::Mat dst=image_padded.clone();
    std::vector<float> dst_data;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(dst, bgrChannels);
    for (auto i = 0; i < bgrChannels.size(); i++)
    {
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
        dst_data.insert(dst_data.end(), data.begin(), data.end());
    }
    FILE* bfp = fopen("image_preprocessed_wxw.bin", "wb");
    fwrite(dst_data.data(), sizeof(float), image_channel * image_width * image_height, bfp);
    fclose(bfp);
}


int main() {
    load_image_and_preprocess();
    return 0;
}
