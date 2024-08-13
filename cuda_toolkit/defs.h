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

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_2D_BLOCK_DIM 32

#define TOTAL_TOBE_CHANNELS 256
#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

#define ABS(x)         ((x) < 0 ? -(x) : (x))
#define NORM_F2(v) (sqrtf( v.x * v.x + v.y * v.y))
#define SCALE_F2(v, a) ({v.x * a, v.y * a});

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





namespace defs
{
	// Matlab strings are u16, even though variable names aren't
	static const std::u16string Plane_tx_name = u"plane";
	static const std::u16string X_line_tx_name = u"xLine";
	static const std::u16string Y_line_tx_name = u"yLine";

	static const char* Rf_data_name = "rx_scans";
	static const char* Loc_data_name = "rx_locs";
	static const char* Tx_config_name = "tx_config";

	static const char* F0_name = "f0";
	static const char* Fs_name = "fs";

	static const char* Column_count_name = "cols";
	static const char* Row_count_name = "rows";
	static const char* Width_name = "width";
	static const char* Pitch_name = "pitch";

	static const char* X_min_name = "x_min";
	static const char* x_max_name = "x_max";
	static const char* Y_min_name = "y_min";
	static const char* Y_max_name = "y_max";

	static const char* Tx_count_name = "no_transmits";
	static const char* Src_location_name = "src";
	static const char* Transmit_type_name = "transmit";
	static const char* Pulse_delay_name = "pulse_delay";


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
		
		float3 src_pos;
		
		TransmitType tx_type;
		ulonglong4 volume_size;
	};


	struct RfDataDims {
		size_t channel_count;
		size_t sample_count;
		size_t tx_count;
	};

	struct ArrayParams {
		float f0; // Transducer frequency (Hz)
		float fs; // Data sample rate (Hz)

		int column_count; // Array column count
		int row_count; // Array row count;
		float width; // Element width (m)
		float pitch; // Element pitch (m)

		float x_min; // Transducer left elements (m)
		float x_max; // Transudcer right elements (m)
		float y_min; // Transducer bottom elements (m)
		float y_max; // Transducer top elements (m)

		int tx_count; // Number of transmittions
		float3 src_location; // Location of tx source (m)
		TransmitType transmit_type; // Transmit type
		float pulse_delay; // Delay to center of pulse (seconds)
	};


	struct MappedFileHandle {
		void* file_handle;
		void* file_view;
	};

}

#endif // !DEFS_H

