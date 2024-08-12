#ifndef BEAMFORMER_CUH
#define	BEAMFORMER_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"

__constant__ defs::KernelConstants Constants;


namespace old_beamformer
{

	bool configure_textures(VolumeConfiguration* config);

	bool beamform(float* d_volume, const cuComplex* d_rf_data, const float* d_loc_data, float3 src_pos);


}

namespace _kernels
{
	__global__ void
	old_complexDelayAndSum(const cuComplex* rfData, const float* locData, float* volume, cudaTextureObject_t textures[3]);

	__device__ float
	f_num_aprodization(float3 vox_loc, float3 element_loc, float f_num);
}
	



#endif // !BEAMFORMER_CUH
