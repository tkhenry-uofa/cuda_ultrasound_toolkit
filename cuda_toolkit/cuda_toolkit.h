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
		typedef unsigned int uint;

		typedef struct {
			float re;
			float im;
		} complex_f;

		typedef struct {
			uint channel_mapping[256];
			uint decoded_dims[3];
			uint raw_dims[2];
			bool rx_cols;
		} BeamformerParams;

		EXPORT_FN bool cleanup();

		EXPORT_FN bool raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols);

		EXPORT_FN bool init_cuda_configuration(const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols);

		EXPORT_FN bool deinit_cuda_configuration();

		EXPORT_FN bool register_cuda_buffers(uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_sso);

		/**
		* Converts input to floats, hadamard decodes, and hilbert transforms via fft
		* 
		* Padding at the end of each channel is skipped, along with the transmitting channels
		* 
		* input_dims - [raw_sample_count (sample_count * tx_count + padding), total_channel_count]	
		* decoded_dims - [sample_count, rx_channel_count, tx_count]
		* rx_cols - TRUE|FALSE: The first|second half of the input channels are read
		*/
		EXPORT_FN bool test_convert_and_decode(const int16_t* input, const BeamformerParams params, complex_f** complex_out, complex_f** intermediate);

		EXPORT_FN bool decode_and_hilbert(size_t input_offset, uint output_buffer);

#ifdef __cplusplus
	}
#endif



#endif // !CUDA_TOOLKIT_H

