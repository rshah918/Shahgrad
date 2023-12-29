#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include "../shahgrad.cpp"
using namespace std;

std::vector<Value*> read_jpg(const std::string& file_path) {
    // Read the black and white JPG file
    cv::Mat image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Unable to read the image." << std::endl;
        return {};  // Return an empty vector in case of an error
    }

    std::vector<Value*> im;
    for (int i = 0; i < image.rows; ++i) {
        const uchar* rowPtr = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; ++j) {
            //normalize pixel between 0-1
            float normalizedPixel = static_cast<float>(rowPtr[j]) / 255.0;
            //convert to a Value node
            im.push_back(new Value(normalizedPixel));
        }
    }

    return im;
}

std::pair<std::vector<std::vector<Value*>>, std::vector<std::vector<float>>> load_data() {
    const int num_classes = 10;  // Number of classes (0-9)
    const int num_images_per_class = 100;  // Number of images per class

    std::vector<std::vector<Value*> > x_train;
    std::vector<vector<float> > y_train;

    for (int class_label = 0; class_label < num_classes; ++class_label) {
        for (int image_index = 0; image_index < num_images_per_class; ++image_index) {
            // Construct the file path
            std::string file_path = "/Users/rahulshah/Documents/Personal Projects/Shahgrad/MNIST/MNIST10x10/" + std::to_string(class_label) + "/" + std::to_string(image_index) + ".jpg";

            // Read image and pixel values
            std::vector<Value*> imageData = read_jpg(file_path);
            x_train.push_back(imageData);

            //populate y_train
            std::vector<float> one_hot_encoded_label(num_classes, 0.0);
            one_hot_encoded_label[class_label] = 1.0;
            y_train.push_back(one_hot_encoded_label);
        }
    }

    // shuffle data randomly
    int index_1 = 0;
    int index_2 = 0;
    for (int i = 0; i < 1000; i++){
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0, 900); // define the range
        //create 2 random numbers
        index_1 = distr(gen);
        index_2 = distr(gen);
        //swap at those indicies
        std::swap(x_train[index_1], x_train[index_2]);
        std::swap(y_train[index_1], y_train[index_2]);
    }

    return std::make_pair(x_train, y_train);
}


Model * createModel(){
    Model * m = new Model(100);
    m->add_layer("linear", 64);
    m->add_layer("linear", 10);
    m->add_layer("softmax", 10);
    return m;
}

int main(){
    Model * m =  createModel();
    cout << "Model Created" << endl;
    // Load data
    cout << "Loading data..." << endl;
    auto data = load_data();
    cout << "Data load complete" << endl;
    
    // Use const_cast to remove the const-qualification
    std::vector<std::vector<Value*> > x_train = data.first;
    std::vector<std::vector<float> > y_train = data.second;

    cout << "Input vector size: "<< x_train[0].size() << endl;
    cout << "Output vector size: "<< y_train[0].size() << endl;
    cout << "Number of training samples: " << x_train.size() << endl;
    cout << "Compiling model" << endl;
    m->compile(x_train[0]);
    cout << "Model shape: " << endl;
    cout <<  m->layers[0]->input_size << endl;
    cout <<  m->layers[0]->output_size << endl;
    cout <<  m->layers[1]->input_size << endl;
    cout <<  m->layers[1]->output_size << endl;
    cout <<  m->layers.back()->output_size << endl;
    m->train(x_train, y_train, 1);
    m->layers.back()->visualizeGraph();
    return 0;
}
/*
Fix OOM error. 
    -alright seems like the visualizer is memory hungry
    -might need a data loader
Visualizer only shows softmax layer???
Alright yea my data loading is wrong. Manually creaing a ~700 long input vector works fine but not through the function

*/
