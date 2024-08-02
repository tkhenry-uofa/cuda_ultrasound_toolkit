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

#include <cufft.h>
#include <cublas_v2.h>



#define MAX_THREADS_PER_BLOCK 1024
#define MAX_2D_BLOCK_DIM 32
#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

typedef unsigned int uint;
typedef int16_t i16;

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

    bool init;

} CudaSession;

extern CudaSession Session;

// CUDA API error checking
#define CUDA_THROW_IF_ERROR(err)                                                                        \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s (%d) '%s'\n At %s:%d\n",                                    \
                cudaGetErrorName(err_), err_, cudaGetErrorString(err_), __FILE__, __LINE__);       \
            throw std::runtime_error("cuda error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)



namespace defs
{
	static const std::string rf_data_name = "rx_scans";

	struct RfDataDims {
        uint sample_count;
		uint channel_count;
		uint tx_count;
	};
}


// TODO: remove


#define MAX_ERROR_LENGTH 256
static char Error_buffer[MAX_ERROR_LENGTH];
#define FFT_RETURN_IF_ERROR(STATUS, MESSAGE)		\
{													\
	strcpy(Error_buffer, MESSAGE);					\
	strcat(Error_buffer, " Error code: %d.\n");		\
	if (STATUS != CUFFT_SUCCESS) {					\
		fprintf(stderr,Error_buffer,(int)STATUS);	\
		return false; }								\
}	

#endif // !DEFS_H

