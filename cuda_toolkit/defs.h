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
#include <complex>

#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_gl_interop.h>

#define PI_F 3.141592654f
#define NaN (float)0xFFFFFFFF; 

#define MAX_THREADS_PER_BLOCK 128
#define MAX_2D_BLOCK_DIM 32

#define WARP_SIZE 32

#define TOTAL_TOBE_CHANNELS 256
#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

#define SCALAR_ABS(x)         ((x) < 0 ? -(x) : (x))
#define SCALE_F2(v, a) {v.x * a, v.y * a};


#define NORM_F2(v) (sqrtf( v.x * v.x + v.y * v.y))
#define NORM_F3(v) (sqrtf( v.x * v.x + v.y * v.y + v.z * v.z))


#define ADD_F2(v,u) {(v).x + (u).x, (v).y + (u).y}

typedef unsigned int uint;
typedef int16_t i16;
typedef std::vector<std::complex<float>> ComplexVectorF;


struct BufferMapping
{
    cudaGraphicsResource_t cuda_resource;
    uint gl_buffer_id;
};

struct VolumeConfiguration
{
	cudaArray_t d_texture_arrays[3] = { nullptr, nullptr, nullptr };
	cudaTextureObject_t textures[3];
	uint3 voxel_counts; // x, y, z
	size_t total_voxels;
	float3 minimums;
	float3 maximums;
	float axial_resolution;
	float lateral_resolution;
};

struct CudaSession
{
    bool init = false;
    bool rx_cols = false;
	uint2 input_dims;
	uint3 decoded_dims;

    cublasHandle_t cublas_handle = nullptr;
    cufftHandle forward_plan;
    cufftHandle inverse_plan;

    int16_t* d_input = nullptr;
    float* d_converted = nullptr;
    float* d_decoded = nullptr;
    cufftComplex* d_complex = nullptr;
    float* d_hadamard = nullptr;

    BufferMapping raw_data_ssbo;
    BufferMapping* rf_data_ssbos = nullptr;
    uint rf_buffer_count;

	VolumeConfiguration volume_configuration;
	
    uint* channel_mapping = nullptr;

	float element_pitch;
};


extern CudaSession Session;


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


struct PositionTextures {
	cudaTextureObject_t x, y, z;
};

enum TransmitType
{
	TX_PLANE = 0,
	TX_X_FOCUS = 1,
	TX_Y_FOCUS = 2,
};

struct KernelConstants
{
	size_t sample_count;
	size_t channel_count;
	size_t tx_count;
	uint3 voxel_dims;
	float3 volume_mins;
	float3 resolutions;
	float3 src_pos;
	TransmitType tx_type;
	float element_pitch;
};


struct MappedFileHandle {
	void* file_handle;
	void* file_view;
};

#ifdef _DEBUG
	#include <assert.h>
	#define ASSERT(x) assert(x)
#else
	#define ASSERT(x)
#endif // _DEBUG


// CUDA API error checking
#define CUDA_RETURN_IF_ERROR(err)                                                            \
    do {                                                                                    \
        cudaError_t err_ = (err);                                                           \
        if (err_ != cudaSuccess) {                                                          \
            std::printf("CUDA error %s (%d) '%s'\n At %s:%d\n",                             \
                cudaGetErrorName(err_), err_, cudaGetErrorString(err_), __FILE__, __LINE__);\
            ASSERT(false);																	\
			return false;																	\
        }                                                                                   \
    } while (0)

// cublas API error checking
#define CUBLAS_THROW_IF_ERR(err)                                                            \
    do {                                                                                    \
        cublasStatus_t err_ = (err);                                                        \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);            \
            ASSERT(false);                                                                  \
			return false;																	\
        }                                                                                   \
    } while (0)

// cufft API error checking
#define CUFFT_THROW_IF_ERR(err)                                                             \
    do {                                                                                    \
        cufftResult_t err_ = (err);                                                         \
        if (err_ != CUFFT_SUCCESS) {                                                        \
            std::printf("cufft error %d at %s:%d\n", err_, __FILE__, __LINE__);             \
            ASSERT(false);                                                                  \
			return false;																	\
		}                                                                                   \
    } while (0)


#define TIME_FUNCTION(CALL, MESSAGE)                                                        \
{                                                                                           \
    auto start = std::chrono::high_resolution_clock::now();                                 \
    CALL;                                                                                   \
    CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());                                           \
    auto elapsed = std::chrono::high_resolution_clock::now() - start;                       \
    std::cout << MESSAGE << " " <<                                                          \
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count() <<         \
        " seconds." << std::endl;                                                           \
}  



#endif // !DEFS_H

