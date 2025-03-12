#ifndef CUDA_TOOLKIT_TESTING_H
#define CUDA_TOOLKIT_TESTING_H

#include <stdint.h>
#include <cuComplex.h>

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
	float pitch[2];
	uint row_count;
	uint col_count;
	float xdc_mins[2];
	float xdc_maxes[2];
} ArrayParams;

typedef enum {
	INT_16 = 0,
	FLOAT_32 = 1
} RfDataType;

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
	u16 channel_mapping[512];

	uint readi_group_id;
	uint readi_group_size;

	RfDataType rf_data_type;

	float f_number;

	int sequence;

} PipelineParams;

/**
* Test functions
*/

EXPORT_FN bool raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const u16* channel_mapping);

/**
* Full pipeline
* 1. Convert to floats
* 2. Decode
* 3. Hilbert
* 4. Beamform
*/
EXPORT_FN bool readi_beamform_raw(const int16_t* input, PipelineParams params, cuComplex** volume);

/**
* Pipeline without conversion
*/
EXPORT_FN bool readi_beamform_fii(const float* input, PipelineParams params, cuComplex** volume);

/**
* Just hilbert and beamform
*/
EXPORT_FN bool fully_sampled_beamform(const float* input, PipelineParams params, cuComplex** volume);



#endif // !CUDA_TOOLKIT_TESTING_H

