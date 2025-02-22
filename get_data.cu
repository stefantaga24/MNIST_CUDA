#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

int bswap(int x)
{
    int b0 = x&255;
    int b1 = (x>>8)&255;
    int b2 = (x>>16)&255;
    int b3 = (x>>24)&255;
    return (b0<<24)+(b1<<16)+(b2<<8)+b3;
}
// Function to read the header of MNIST files
void read_header(std::ifstream &file, int &magic_number, int &num_items, int &rows, int &cols) {
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Convert from big-endian to little-endian if necessary
    magic_number = bswap(magic_number);
    num_items = bswap(num_items);
    rows = bswap(rows);
    cols = bswap(cols);
}

// Function to read MNIST images
std::vector<std::vector<std::vector<unsigned char>>> read_images(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0, num_images=0, rows=0, cols=0;
    read_header(file, magic_number, num_images, rows, cols);

    // Read the image data
    std::vector<std::vector<std::vector<unsigned char>>> images(num_images, std::vector<std::vector<unsigned char>>(rows, std::vector<unsigned char>(cols)));
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                file.read(reinterpret_cast<char*>(&images[i][r][c]), 1);
            }
        }
    }

    return images;
}

// Function to read MNIST labels
std::vector<unsigned char> read_labels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0, num_labels = 0;
    int rows = 0, cols = 0;
    read_header(file, magic_number, num_labels, rows, cols);  // Only need num_labels from header

    // Read the label data
    std::vector<unsigned char> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        file.read(reinterpret_cast<char*>(&labels[i]), 1);
    }

    return labels;
}


// This can be optimized
#define TILE_WIDTH 16 
__global__ void ConvLayerForward_Kernel(int M, int C, int H_grid, int W_grid,
                                        int K, float* X, float* W, float* Y,
                                        int H_input, int W_input)
{
    int m = blockIdx.x; // What filter bank I am in 
    int h = (blockIdx.y/W_grid)*TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y%W_grid)*TILE_WIDTH + threadIdx.x; 
    int n = blockIdx.z; 
    float acc = 0.;
    for (int c=0;c<C;c++)
    {
        for (int p=0;p<K;p++)
        {
            for (int q=0;q<K;q++)
            {
                acc += X[n*C*H_input*W_input + c*H_input*W_input + (h+p)*W_input + w+q] * 
                       W[m * C * K * K + c * K * K + p * K + q];
            }
        }
    }
    int H_output = H_input - K + 1;
    int W_output = W_output - K + 1;
    Y[n*M*C*H_output*W_output + m*C*H_output*W_output + h * W_output + w] = acc;
}


bool test_conv_batch()
{
    return True;
}
int main() {
    try {
        // Read images and labels
        auto images = read_images("MNIST_database/train-images.idx3-ubyte");
        auto labels = read_labels("MNIST_database/train-labels.idx1-ubyte");

        std::cout << "Number of images: " << images.size() << std::endl;
        std::cout << "Number of labels: " << labels.size() << std::endl;

        // Access the first image and its label
        std::cout << "First image dimensions: " << images[0].size() << "x" << images[0][0].size() << std::endl;
        std::cout << "First label: " << (int)labels[0] << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
