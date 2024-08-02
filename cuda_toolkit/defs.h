#ifndef DEFS_H
#define DEFS_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include<cuda_gl_interop.h>

#include <stdexcept>
#include <stdio.h>
#include <string>
#include <string.h>
#include <chrono>
#include <vector>

#include <cufft.h>
#include <cublas_v2.h>

#include <assert.h>



#define MAX_THREADS_PER_BLOCK 1024
#define MAX_2D_BLOCK_DIM 32
#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

typedef unsigned int uint;
typedef int16_t i16;

typedef struct
{
    cudaGraphicsResource_t cuda_resource;
    uint gl_buffer_id;
} buffer_mapping;

typedef struct
{
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

    buffer_mapping* buffers;
    uint buffer_count;

    bool init;

} CudaSession;

extern CudaSession Session;

#define TIME_FUNCTION(CALL, MESSAGE)                                                        \
{                                                                                           \
    auto start = std::chrono::high_resolution_clock::now();                                 \
    CALL;                                                                                   \
    auto elapsed = std::chrono::high_resolution_clock::now() - start;                       \
    std::cout << MESSAGE << " " <<                                                          \
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count() <<         \
        " seconds." << std::endl;                                                           \
}                                                                                                   

// CUDA API error checking
#define CUDA_THROW_IF_ERROR(err)                                                             \
    do {                                                                                    \
        cudaError_t err_ = (err);                                                           \
        if (err_ != cudaSuccess) {                                                          \
            std::printf("CUDA error %s (%d) '%s'\n At %s:%d\n",                             \
                cudaGetErrorName(err_), err_, cudaGetErrorString(err_), __FILE__, __LINE__);\
            assert(false);                                                                   \
        }                                                                                   \
    } while (0)

// cublas API error checking
#define CUBLAS_THROW_IF_ERR(err)                                                           \
    do {                                                                                    \
        cublasStatus_t err_ = (err);                                                        \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);            \
            assert(false);                                                                   \
        }                                                                                   \
    } while (0)

// cufft API error checking
#define CUFFT_THROW_IF_ERR(err)                                                            \
    do {                                                                                    \
        cufftResult_t err_ = (err);                                                         \
        if (err_ != CUFFT_SUCCESS) {                                                        \
            std::printf("cufft error %d at %s:%d\n", err_, __FILE__, __LINE__);             \
            assert(false);                                                                   \
        }                                                                                   \
    } while (0)



namespace defs
{
	static const std::string rf_data_name = "rx_scans";

    // Annotating Dim order
    typedef union
    {
        struct
        {
            uint sample_count;
            uint channel_count;
            uint tx_count;
        };
        struct uint3;

    } RfDataDims;
    
}


#endif // !DEFS_H

