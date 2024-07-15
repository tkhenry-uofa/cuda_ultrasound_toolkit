#ifndef KERNEL_CUH
#define KERNEL_CU

#include <vector>
#include <complex>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"


bool cuda_fft(const std::vector<float>& real_data, std::vector<std::complex<float>>** im_out, defs::RfDataDims dims);


#endif // !KERNEL_CUH
