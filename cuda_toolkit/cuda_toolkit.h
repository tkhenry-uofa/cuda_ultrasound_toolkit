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

		EXPORT_FN bool unregister_cuda_buffers();

		EXPORT_FN bool deinit_cuda_configuration();

		EXPORT_FN bool init_cuda_configuration(const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols);

		EXPORT_FN bool register_cuda_buffers(uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_sso);

		EXPORT_FN bool decode_and_hilbert(size_t input_offset, uint output_buffer);

		bool init_session(const uint input_dims[2], const uint decoded_dims[3], const uint channel_mapping[256], bool rx_cols);


#ifdef __cplusplus
	}
#endif

#endif // !CUDA_TOOLKIT_H

