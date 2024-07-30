#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <vector>
#include <complex>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"


bool cuda_fft(const float* input, defs::ComplexF** output, defs::RfDataDims dims);


#endif // !KERNEL_CUH
