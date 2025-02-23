#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>

#define eps 1e-3 // Probably could do some arithmetic tricks for better precision
int bswap(int x)
{
    int b0 = x & 255;
    int b1 = (x >> 8) & 255;
    int b2 = (x >> 16) & 255;
    int b3 = (x >> 24) & 255;
    return (b0 << 24) + (b1 << 16) + (b2 << 8) + b3;
}
// Function to read the header of MNIST files
void read_header(std::ifstream &file, int &magic_number, int &num_items, int &rows, int &cols)
{
    file.read(reinterpret_cast<char *>(&magic_number), 4);
    file.read(reinterpret_cast<char *>(&num_items), 4);
    file.read(reinterpret_cast<char *>(&rows), 4);
    file.read(reinterpret_cast<char *>(&cols), 4);

    // Convert from big-endian to little-endian if necessary
    magic_number = bswap(magic_number);
    num_items = bswap(num_items);
    rows = bswap(rows);
    cols = bswap(cols);
}

// Function to read MNIST images
std::vector<std::vector<std::vector<unsigned char>>> read_images(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    read_header(file, magic_number, num_images, rows, cols);

    // Read the image data
    std::vector<std::vector<std::vector<unsigned char>>> images(num_images, std::vector<std::vector<unsigned char>>(rows, std::vector<unsigned char>(cols)));
    for (int i = 0; i < num_images; ++i)
    {
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                file.read(reinterpret_cast<char *>(&images[i][r][c]), 1);
            }
        }
    }

    return images;
}

// Function to read MNIST labels
std::vector<unsigned char> read_labels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int magic_number = 0, num_labels = 0;
    int rows = 0, cols = 0;
    read_header(file, magic_number, num_labels, rows, cols); // Only need num_labels from header

    // Read the label data
    std::vector<unsigned char> labels(num_labels);
    for (int i = 0; i < num_labels; ++i)
    {
        file.read(reinterpret_cast<char *>(&labels[i]), 1);
    }

    return labels;
}

// This can be optimized
#define TILE_WIDTH 16
__global__ void ConvLayerForward_Kernel(int M, int C, int H_grid, int W_grid,
                                        int K, float *X, float *W, float *Y,
                                        int H_input, int W_input,
                                        int Y_size, int X_size)
{
    int m = blockIdx.x; // What filter bank I am in
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int n = blockIdx.z;
    int H_output = H_input - K + 1;
    int W_output = W_input - K + 1;
    int y_pos = n * M * H_output * W_output + m * H_output * W_output + h * W_output + w;
    float acc = 0.;
    if (0 <= h && h < H_output && 0 <= w && w < W_output)
    {
        for (int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    int x_idx = n * C * H_input * W_input + c * H_input * W_input + (h + p) * W_input + w + q;
                    if (0 <= x_idx && x_idx < X_size)
                    {
                        acc += X[x_idx]* W[m * C * K * K + c * K * K + p * K + q];
                    }
                }
            }
        }
        Y[y_pos] = acc;
    }
}

void convLayer_batched(int N, int M, int C, int H_X, int W_X, int K, float *X, float *W, float *Y)
{
    int H_out = H_X - K + 1;
    int W_out = W_X - K + 1;
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int h = 0; h < H_out; h++)
            {
                for (int w = 0; w < W_out; w++)
                {
                    int y_idx = n * M * H_out * W_out + m * H_out * W_out + h * W_out + w;
                    Y[y_idx] = 0;
                    for (int c = 0; c < C; c++)
                    {
                        for (int p = 0; p < K; p++)
                        {
                            for (int q = 0; q < K; q++)
                            {
                                int x_idx = n * C * H_X * W_X + c * H_X * W_X + (h + p) * W_X + (w + q);
                                int w_idx = m * C * K * K + c * K * K + p * K + q;
                                Y[y_idx] = Y[y_idx] + X[x_idx] * W[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

bool test_conv_batch()
{
    int N = 1, M = 1, H_X = 100, W_X = 100, C = 1, K = 5;
    int H_out = H_X - K + 1, W_out = W_X - K + 1;
    // Random number generator setup
    std::random_device rd;                                  // Seed for random number engine
    std::mt19937 gen(rd());                                 // Mersenne Twister PRNG
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Range [0,1]

    unsigned long long size_X = N * C * H_X * W_X;
    unsigned long long size_W = M * C * K * K;
    unsigned long long size_Y = N * M * H_out * W_out;
    float *X = new float[size_X]; // It is a tensor of shape [N,C,H_X,W_X]
    float *W = new float[size_W]; // It is a tensor of shape [M,C,K,K], also called filter banks but they're the weights
    float *Y = new float[size_Y]; // It is a tensor of the shape [N,M,H_out,W_out]
    float *Y_cuda_result = new float[size_Y];
    for (int i = 0; i < size_X; i++)
    {
        X[i] = dist(gen);
    }
    for (int i = 0; i < size_W; i++)
    {
        W[i] = dist(gen);
    }
    for (int i = 0; i < size_Y; i++)
    {
        Y[i] = 0;
        Y_cuda_result[i] = 0;
    }
    float *d_X, *d_W, *d_Y;
    cudaMalloc((void **)&d_X, sizeof(float) * size_X);
    cudaMalloc((void **)&d_W, sizeof(float) * size_W);
    cudaMalloc((void **)&d_Y, sizeof(float) * size_Y);

    cudaMemcpy(d_X, X, sizeof(float) * size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, sizeof(float) * size_W, cudaMemcpyHostToDevice);

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int T = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, T, N);
    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(M, C, H_grid, W_grid, K, d_X,
                                                   d_W, d_Y, H_X, W_X,
                                                   size_Y, size_X);
    cudaMemcpy(Y_cuda_result, d_Y, sizeof(float) * size_Y, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    convLayer_batched(N, M, C, H_X, W_X, K, X, W, Y);

    bool test_correct = true;
    int number_results = 0;
    for (int i = 0; i < size_Y; i++)
    {
        if (abs(Y[i] - Y_cuda_result[i]) > eps)
        {
            std::cout << "TEST convolution has failed on position i = " << i << " " << "Y[i] = " << Y[i]
                      << " " << "Y_cuda_result[i] = " << Y_cuda_result[i] << '\n';
            test_correct = false;
            number_results = number_results + 1;
        }
    }
    printf("Number of wrong results: %d \n", number_results);
    delete[] Y;
    Y = nullptr;
    delete[] Y_cuda_result;
    Y_cuda_result = nullptr;
    delete[] X;
    X = nullptr;
    delete[] W;
    W = nullptr;
    if (test_correct == true)
    {
        printf("INFO >>> Convolution test passed!\n");
    }
    return test_correct;
}

void subsamplingLayer_forward(int N, int M, int H,
                              int W, int K, float *X, float *S, float *b)
{
    int H_result = H / K;
    int W_result = W / K;
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int h = 0; h < H_result; h++)
            {
                for (int w = 0; w < W_result; w++)
                {
                    size_t s_idx = n * M * H_result * W_result + m * H_result * W_result + h * W_result + w;
                    S[s_idx] = 0.;
                    for (int p = 0; p < K; p++)
                    {
                        for (int q = 0; q < K; q++)
                        {
                            S[s_idx] += (X[n * M * H * W + m * H * W + (h+p) * W + w+q] / (K * K));
                        }
                    }
                    S[s_idx] = S[s_idx] + b[n * M + m] > 0 ? S[s_idx] + b[n * M + m] : 0;
                }
            }
        }
    }
}

#define TILE_WIDTH 16
__global__ void SubSamplingLayer_forward(int M, int W_grid,
                                         int K, float *X, float *b, float *Y,
                                         int H_input, int W_input)
{
    int m = blockIdx.x; // What filter bank I am in
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int n = blockIdx.z;
    int H_output = H_input / K;
    int W_output = W_input / K;
    int y_pos = n * M * H_output * W_output + m * H_output * W_output + h * W_output + w;
    float acc = 0.;
    if (0 <= m && m < M && 0 <= h && h < H_output && 0 <= w && w < H_output)
    {
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                int x_idx = n * M * H_input * W_input + m * H_input * W_input + (h + p) * W_input + w + q;
                acc = acc + (X[x_idx] / (K * K)); // Is this more numerically stable
            }
        }
        Y[y_pos] = acc + b[n * M + m] > 0 ? acc + b[n * M + m] : 0;
    }
}

bool test_subsampling_batch()
{
    int N = 3, M = 5, H_X = 16, W_X = 16, K = 2;
    int H_out = H_X / K, W_out = W_X / K;
    // Random number generator setup
    std::random_device rd;                                  // Seed for random number engine
    std::mt19937 gen(rd());                                 // Mersenne Twister PRNG
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Range [0,1]

    unsigned long long size_X = N * M * H_X * W_X;
    unsigned long long size_Y = N * M * H_out * W_out;
    size_t size_b = N * M;
    float *X = new float[size_X]; // It is a tensor of shape [N,M,H_X,W_X]
    float *Y = new float[size_Y]; // It is a tensor of the shape [N,M,H_out,W_out]
    float *b = new float[N * M];
    float *Y_cuda_result = new float[size_Y];
    for (int i = 0; i < size_X; i++)
    {
        X[i] = dist(gen);
    }
    for (int i = 0; i < size_b; i++)
    {
        b[i] = 0;
    }
    float *d_X, *d_Y, *d_b;
    cudaMalloc((void **)&d_X, sizeof(float) * size_X);
    cudaMalloc((void **)&d_Y, sizeof(float) * size_Y);
    cudaMalloc((void **)&d_b, sizeof(float) * size_b);
    cudaMemcpy(d_X, X, sizeof(float) * size_X, cudaMemcpyHostToDevice);

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int T = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, T, N);
    SubSamplingLayer_forward<<<gridDim, blockDim>>>(M, W_grid, K, d_X,
                                                    d_b, d_Y, H_X, W_X);
    cudaMemcpy(Y_cuda_result, d_Y, sizeof(float) * size_Y, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_b);

    subsamplingLayer_forward(N, M, H_X, W_X, K, X, Y, b);

    bool test_correct = true;
    int number_results = 0;
    for (int i = 0; i < size_Y; i++)
    {
        if (abs(Y[i] - Y_cuda_result[i]) > eps)
        {
            std::cout << "TEST subsampling has failed on position i = " << i << " " << "Y[i] = " << Y[i]
                      << " " << "Y_cuda_result[i] = " << Y_cuda_result[i] << '\n';
            test_correct = false;
            number_results = number_results + 1;
        }
    }
    printf("Number of wrong results: %d \n", number_results);
    delete[] Y;
    Y = nullptr;
    delete[] Y_cuda_result;
    Y_cuda_result = nullptr;
    delete[] X;
    X = nullptr;
    delete[] b;
    b = nullptr;
    if (test_correct == true)
    {
        printf("INFO >>> Convolution test passed!\n");
    }
    return test_correct;
}
int main()
{
    try
    {
        /*
        // Read images and labels
        auto images = read_images("MNIST_database/train-images.idx3-ubyte");
        auto labels = read_labels("MNIST_database/train-labels.idx1-ubyte");

        std::cout << "Number of images: " << images.size() << std::endl;
        std::cout << "Number of labels: " << labels.size() << std::endl;

        // Access the first image and its label
        std::cout << "First image dimensions: " << images[0].size() << "x" << images[0][0].size() << std::endl;
        std::cout << "First label: " << (int)labels[0] << std::endl;
        */
        test_conv_batch();
        test_subsampling_batch();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
