#ifndef CUDA_TOOLKIT_H
#define CUDA_TOOLKIT_H

#include <stdint.h>

#ifdef _WIN32
	#define EXPORT_FN _declspec(dllexport)
#else
	#define EXPORT_FN
#endif

#ifdef __cplusplus
	extern "C" {
#endif

		typedef enum {
			SUCCESS = 0,
			FAILURE = 1,
		} result_t;

		typedef struct {
			float re;
			float im;
		} complex_f;

		/**
		* Calculates a batch of un-normalizeed hilbert transform. 
		* cuFFT limits all dimensions to i32 for single GPUs 
		* 
		* sample_count - Number of samples per channel
		* channel_count - Number of channels
		* input - Input NxM float array, M channels of N samples each
		* output - Complex output, same dimensions as the input or NULL if failure
		*		   Caller is responsible for cleanup
		*/
		EXPORT_FN result_t batch_hilbert_transform(int sample_count, int channel_count, const float* input, complex_f** output);

#ifdef __cplusplus
	}
#endif



#endif // !CUDA_TOOLKIT_H

