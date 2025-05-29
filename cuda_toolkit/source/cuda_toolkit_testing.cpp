#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"
#include "beamformer/beamformer.cuh"

#include "cuda_toolkit_testing.h"


bool readi_beamform_raw(const int16_t* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	uint2 raw_data_dims = *(uint2*)&(params.raw_dims);
	size_t total_count = (size_t)dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;
	size_t total_raw_count = (size_t)raw_data_dims.x * raw_data_dims.y;

	init_cuda_configuration(params.raw_dims, params.decoded_dims);
	cuda_set_channel_mapping(params.channel_mapping);

	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	vol_config.voxel_counts = *(uint3*)&params.vol_counts;
	vol_config.total_voxels = (size_t)vol_config.voxel_counts.x * vol_config.voxel_counts.y * vol_config.voxel_counts.z;

	Sessions.volume_configuration = vol_config;

	Sessions.pitches.x = params.array_params.pitch[0];
	Sessions.pitches.y = params.array_params.pitch[1];

	Sessions.xdc_mins.x = params.array_params.xdc_mins[0];
	Sessions.xdc_mins.y = params.array_params.xdc_mins[1];

	Sessions.xdc_maxes.x = params.array_params.xdc_maxes[0];
	Sessions.xdc_maxes.y = params.array_params.xdc_maxes[1];

	Sessions.sequence = (TransmitModes)params.sequence;

	Sessions.pulse_delay = params.pulse_delay;

	Sessions.mixes_count = params.mixes_count;
	Sessions.mixes_offset = params.mixes_offset;
	memcpy(Sessions.mixes_rows, params.mixes_rows, sizeof(params.mixes_rows));

	i16* d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_raw_count * sizeof(i16)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input, total_raw_count * sizeof(i16), cudaMemcpyHostToDevice));

	// Recreate the fft plans for readi subgroup sized batches
	// TODO: Do this in a better way
	cufftDestroy(Sessions.forward_plan);
	cufftDestroy(Sessions.inverse_plan);
	cufftDestroy(Sessions.strided_plan);

	if (params.filter_length != 0)
	{
		cuda_set_match_filter(params.match_filter, params.filter_length);
	}

	hilbert::plan_hilbert((int)Sessions.decoded_dims.x, (int)Sessions.decoded_dims.y * (int)Sessions.decoded_dims.z);

	i16_to_f::convert_data(d_input, Sessions.d_converted);

	Sessions.readi_group = params.readi_group_id;
	Sessions.readi_group_size = params.readi_group_size;

	hadamard::readi_decode(Sessions.d_converted, Sessions.d_decoded, params.readi_group_id, params.readi_group_size);

	bool do_hilbert = true;
	if (do_hilbert)
	{
		hilbert::hilbert_transform_r2c(Sessions.d_decoded, Sessions.d_complex);
	}
	else
	{
		CUDA_RETURN_IF_ERROR(cudaMemset(Sessions.d_complex, 0x00, total_count * sizeof(cuComplex)));
		CUDA_RETURN_IF_ERROR(cudaMemcpy2D(Sessions.d_complex, 2 * sizeof(float), Sessions.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));
	}

	cuComplex* d_volume;

	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(cuComplex)));

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	int delay_samples = (int)round(params.pulse_delay * params.array_params.sample_freq);
	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(cuComplex));

	std::cout << "Starting beamform\n";

	beamformer::beamform(d_volume, Sessions.d_complex, *(float3*)params.focus, samples_per_meter, params.f_number, delay_samples);


	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(cuComplex), cudaMemcpyDefault));

	cudaFree(d_input);
	cudaFree(d_volume);

	deinit_cuda_configuration();

	return true;
}

bool readi_beamform_fii(const float* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	uint2 raw_data_dims = *(uint2*)&(params.raw_dims);
	size_t total_count = (size_t)dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;
	size_t total_raw_count = (size_t)raw_data_dims.x * raw_data_dims.y;


	init_cuda_configuration(params.raw_dims, params.decoded_dims);
	cuda_set_channel_mapping(params.channel_mapping);

	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[ 2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	vol_config.voxel_counts = *(uint3*) & params.vol_counts;
	vol_config.total_voxels = (size_t)vol_config.voxel_counts.x * vol_config.voxel_counts.y * vol_config.voxel_counts.z;
	Sessions.volume_configuration = vol_config;

	Sessions.pitches.x = params.array_params.pitch[0];
	Sessions.pitches.y = params.array_params.pitch[1];

	Sessions.xdc_mins.x = params.array_params.xdc_mins[0];
	Sessions.xdc_mins.y = params.array_params.xdc_mins[1];	
	
	Sessions.xdc_maxes.x = params.array_params.xdc_maxes[0];
	Sessions.xdc_maxes.y = params.array_params.xdc_maxes[1];

	Sessions.readi_group = params.readi_group_id;
	Sessions.readi_group_size = params.readi_group_size;

	Sessions.sequence = (TransmitModes)params.sequence;

	float *d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_raw_count * sizeof(float)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input, total_raw_count * sizeof(float), cudaMemcpyHostToDevice));

	hadamard::readi_decode(d_input, Sessions.d_decoded, params.readi_group_id, params.readi_group_size);

	//hadamard::readi_staggered_decode(d_input, Sessions.d_decoded, Sessions.d_hadamard);

	//CUDA_RETURN_IF_ERROR(cudaMemcpy(Sessions.d_decoded, d_input, total_raw_count * sizeof(float), cudaMemcpyDeviceToDevice));

	bool do_hilbert = true;
	if (do_hilbert)
	{
		hilbert::hilbert_transform_r2c(Sessions.d_decoded, Sessions.d_complex);
	}
	else
	{
		CUDA_RETURN_IF_ERROR(cudaMemset(Sessions.d_complex, 0x00, total_count * sizeof(cuComplex)));
		CUDA_RETURN_IF_ERROR(cudaMemcpy2D(Sessions.d_complex, 2 * sizeof(float), Sessions.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));
	}
	

	cuComplex* d_volume;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(cuComplex)));
	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(cuComplex));


	uint readi_group_count = dec_data_dims.z / Sessions.readi_group_size;
	//Sessions.pitches.y = Sessions.pitches.y * (float)readi_group_count;

	//Sessions.decoded_dims.z = Sessions.readi_group_size;

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	std::cout << "Starting beamform" << std::endl;
	int delay_samples = (int)round(params.pulse_delay * params.array_params.sample_freq);
	beamformer::beamform(d_volume, Sessions.d_complex, *(float3*)params.focus, samples_per_meter, params.f_number, delay_samples);

	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(cuComplex), cudaMemcpyDefault));

	cudaFree(d_input);
	cudaFree(d_volume);

	deinit_cuda_configuration();

	return true;
}


bool 
beamform(const void* data, uint data_size, CudaBeamformerParameters* bp, void* output)
{
	return true;
}