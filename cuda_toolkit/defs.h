#ifndef DEFS_H
#define DEFS_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include <assert.h>

#include <string>
#include <chrono>
#include <vector>

#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_gl_interop.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_2D_BLOCK_DIM 32

#define TOTAL_TOBE_CHANNELS 256
#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

typedef unsigned int uint;
typedef int16_t i16;


struct BufferMapping
{
    cudaGraphicsResource_t cuda_resource;
    uint gl_buffer_id;
};

struct CudaSession
{

    bool init;
    bool rx_cols;
    uint2 input_dims;
    uint3 decoded_dims;

    cublasHandle_t cublas_handle;
    cufftHandle forward_plan;
    cufftHandle inverse_plan;

    int16_t* d_input;
    float* d_converted;
    float* d_decoded;
    cufftComplex* d_complex;
    float* d_hadamard;

    BufferMapping raw_data_ssbo;
    BufferMapping* rf_data_ssbos;
    uint rf_buffer_count;

    uint* channel_mapping;
};

struct RfDataDims
{
    unsigned int sample_count;
    unsigned int channel_count;
    unsigned int tx_count;

    RfDataDims(const uint3& input)
        : sample_count(input.x), channel_count(input.y), tx_count(input.z) {};

    RfDataDims(const unsigned int* input)
        : sample_count(input[0]), channel_count(input[1]), tx_count(input[2]) {};

    RfDataDims& operator=(const uint3& input)
    {
        sample_count = input.x;
        channel_count = input.y;
        tx_count = input.z;
        return *this;
    }
};

extern CudaSession Session;                                                                                              

// CUDA API error checking
#define CUDA_THROW_IF_ERROR(err)                                                            \
    do {                                                                                    \
        cudaError_t err_ = (err);                                                           \
        if (err_ != cudaSuccess) {                                                          \
            std::printf("CUDA error %s (%d) '%s'\n At %s:%d\n",                             \
                cudaGetErrorName(err_), err_, cudaGetErrorString(err_), __FILE__, __LINE__);\
            assert(false);                                                                  \
        }                                                                                   \
    } while (0)

// cublas API error checking
#define CUBLAS_THROW_IF_ERR(err)                                                            \
    do {                                                                                    \
        cublasStatus_t err_ = (err);                                                        \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);            \
            assert(false);                                                                  \
        }                                                                                   \
    } while (0)

// cufft API error checking
#define CUFFT_THROW_IF_ERR(err)                                                             \
    do {                                                                                    \
        cufftResult_t err_ = (err);                                                         \
        if (err_ != CUFFT_SUCCESS) {                                                        \
            std::printf("cufft error %d at %s:%d\n", err_, __FILE__, __LINE__);             \
            assert(false);                                                                  \
        }                                                                                   \
    } while (0)


#define TIME_FUNCTION(CALL, MESSAGE)                                                        \
{                                                                                           \
    auto start = std::chrono::high_resolution_clock::now();                                 \
    CALL;                                                                                   \
    CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());                                           \
    auto elapsed = std::chrono::high_resolution_clock::now() - start;                       \
    std::cout << MESSAGE << " " <<                                                          \
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count() <<         \
        " seconds." << std::endl;                                                           \
}  

#endif // !DEFS_H

