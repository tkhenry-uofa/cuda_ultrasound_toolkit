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


#endif // !HILBERT_TRANSFORM_CUH
