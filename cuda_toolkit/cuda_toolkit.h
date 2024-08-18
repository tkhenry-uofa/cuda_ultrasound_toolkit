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

		/**
		* API Usage:
		* In any order:
		*	a. When data dimensions are known call init_cuda_configuration to configure the kernel sizes
		*	b. When OGL input and decoded data buffers are created register with register_cuda_buffers
		* Both functions can be indipendently called if configuration changes to handle updates
		*	
		* After init call decode_and_hilbert with the byte offset of the input data in the raw buffer
		* and the index of the output buffer
		*/

		/**
		* input_dims: (samples * transmissions + padding) x raw channels
		* output_dims: samples x channels x transmissions
		* channel_mapping: VSX channels to row-column conversion
		* rx_cols: use the column half of the channel data, otherwise use the row half
		*/
		EXPORT_FN bool init_cuda_configuration(const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols);
		
		/**
		* rf_data_ssbos: List of opengl buffer ids for decoded rf data
		* rf_buffer_count: List length
		* raw_data_ssbo: Opengl buffer id for the raw data buffer
		*/
		EXPORT_FN bool register_cuda_buffers(uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo);

		/**
		* input_offset: Offset into raw_data_ssbo 
		* output_buffer_idx: index into rf_data_ssbos for the output buffer
		*/
		EXPORT_FN bool decode_and_hilbert(size_t input_offset, uint output_buffer_idx);

		// Internal init
		bool _init_session(const uint input_dims[2], const uint decoded_dims[3], const uint channel_mapping[256], bool rx_cols);

#ifdef __cplusplus
	}
#endif

#endif // !CUDA_TOOLKIT_H

