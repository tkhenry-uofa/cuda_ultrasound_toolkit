#ifndef BEAMFORMER_CUH
#define	BEAMFORMER_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../defs.h"

__constant__ defs::KernelConstants Constants;


namespace old_beamformer
{

	bool configure_textures(VolumeConfiguration* config);

	bool beamform(float* d_volume, const cuComplex* d_rf_data, const float2* d_loc_data, const float3 focus_pos, float samples_per_meter);

}

namespace _kernels
{
	__global__ void
	old_complexDelayAndSum(const cuComplex* rfData, const float2* locData, float* volume, cudaTextureObject_t textures[3], float samples_per_meter);

	__device__  __inline__ float
	f_num_aprodization(float3 vox_loc, float2 element_loc, float f_num);
}
	



#endif // !BEAMFORMER_CUH
