#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <string>
#include <string.h>
#include <cufft.h>
#include <cublas_v2.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_2D_BLOCK_DIM 32

#define SAMPLE_F = 50000000 // 50 MHz

#define MAX_ERROR_LENGTH 256
static char Error_buffer[MAX_ERROR_LENGTH];


typedef unsigned int uint;

#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

#define FFT_RETURN_IF_ERROR(STATUS, MESSAGE)		\
{													\
	strcpy(Error_buffer, MESSAGE);					\
	strcat(Error_buffer, " Error code: %d.\n");		\
	if (STATUS != CUFFT_SUCCESS) {					\
		fprintf(stderr,Error_buffer,(int)STATUS);	\
		return false; }								\
}													\

// CUDA API error checking
#define THROW_IF_ERROR(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s (%d) '%s'\n At %s:%d\n",                                    \
                cudaGetErrorName(err_), err_, cudaGetErrorString(err_), __FILE__, __LINE__);       \
            throw std::runtime_error("cuda error");                                                                          \
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
		size_t element_count;
		size_t sample_count;
		size_t tx_count;
	};



}


#endif // !DEFS_H
