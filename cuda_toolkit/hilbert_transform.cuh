#ifndef HILBERT_TRANSFORM_CUH
#define HILBERT_TRANSFORM_CUH

#include <iostream>
#include <vector>
#include <complex>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"


__host__
bool hilbert_transform(int sample_count, int channel_count, const float* input, std::complex<float>** output);

__host__
bool hadamard_decode_cuda(int sample_count, int channel_count, int tx_count, const int* input, float** output);

__host__
void print_array(float* out_array, uint size);


#endif // !HILBERT_TRANSFORM_CUH
