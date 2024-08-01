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

		EXPORT_FN result_t convert_and_decode(const int16_t* input, unsigned int input_dims[2], unsigned int decoded_dims[3], bool rx_rows, float** output);

#ifdef __cplusplus
	}
#endif



#endif // !CUDA_TOOLKIT_H

