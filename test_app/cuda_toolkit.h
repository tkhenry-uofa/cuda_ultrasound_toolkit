#ifndef CUDA_TOOLKIT_H
#define CUDA_TOOLKIT_H

#include <stdint.h>

#ifdef _WIN32
#define EXPORT_FN __declspec(dllexport)
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

	EXPORT_FN bool startup();
	EXPORT_FN bool cleanup();

	EXPORT_FN bool raw_data_to_cuda(const int16_t* input, uint32_t* input_dims, uint32_t* decoded_dims);

	/**
	* Converts input to floats, hadamard decodes, and hilbert transforms via fft
	*
	* Padding at the end of each channel is skipped, along with the transmitting channels
	*
	* input_dims - [raw_sample_count (sample_count * tx_count + padding), total_channel_count]
	* decoded_dims - [sample_count, rx_channel_count, tx_count]
	* rx_rows - TRUE|FALSE: The first|second half of the input channels are read
	*/
	EXPORT_FN result_t test_convert_and_decode(const int16_t* input, uint32_t* input_dims, uint32_t* decoded_dims, bool rx_rows, float** output);

	EXPORT_FN result_t decode_and_hilbert(bool rx_rows, uint32_t output_buffer);

#ifdef __cplusplus
}
#endif



#endif // !CUDA_TOOLKIT_H

