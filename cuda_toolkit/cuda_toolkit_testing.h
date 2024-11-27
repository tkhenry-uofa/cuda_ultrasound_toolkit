#ifndef CUDA_TOOLKIT_TESTING_H
#define CUDA_TOOLKIT_TESTING_H

#include <stdint.h>

#include "cuda_toolkit.h"


typedef unsigned int uint;

typedef struct {
	float re;
	float im;
} complex_f;

typedef struct {
	float center_freq;
	float sample_freq;
	float c;
	float pitch;
	uint row_count;
	uint col_count;
	float xdc_mins[2];
	float xdc_maxes[2];
} ArrayParams;


typedef struct {
	float focus[3];
	float pulse_delay; // Delay to the middle of the pulse(s)
	// x: Sample count
	// y: Rx Channel count
	// z: Acquisition count
	uint decoded_dims[3];

	// x: Sample count x Acquisition count + padding
	// y: Full channel count
	uint raw_dims[2];

	// Volume configuration ( all in meters)
	float vol_mins[3];
	float vol_maxes[3];

	uint vol_counts[3];

	float vol_resolutions[3];

	ArrayParams array_params;

	// Mapping verasonics channels to row and column numbers
	// First half are rows, second half are columns
	u16 channel_mapping[256];


	// Where in the rf data to start reading
	u16 channel_offset;
} PipelineParams;

/**
* Test functions
*/

EXPORT_FN bool raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping);

/**
* Converts input to floats, hadamard decodes, and hilbert transforms via fft
* 
* Padding at the end of each channel is skipped, along with the transmitting channels
* 
* input_dims - [raw_sample_count (sample_count * tx_count + padding), total_channel_count]	
* decoded_dims - [sample_count, rx_channel_count, tx_count]
* rx_cols - TRUE|FALSE: The first|second half of the input channels are read
*/
EXPORT_FN bool test_convert_and_decode(const int16_t* input, const PipelineParams params, complex_f** complex_out, complex_f** intermediate);


EXPORT_FN bool hero_raw_to_beamform(const int16_t* input, PipelineParams params, float** volume);

EXPORT_FN bool readi_beamform_raw(const int16_t* input, PipelineParams params, float** volume);

EXPORT_FN bool fully_sampled_beamform(const float* input, PipelineParams params, float** volume);



#endif // !CUDA_TOOLKIT_TESTING_H

