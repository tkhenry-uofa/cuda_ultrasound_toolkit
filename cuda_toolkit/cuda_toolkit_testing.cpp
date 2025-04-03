#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"
#include "beamformer/beamformer.cuh"

#include "cuda_toolkit_testing.h"


bool
raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const u16* channel_mapping )
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		init_cuda_configuration(input_dims, decoded_dims, channel_mapping);
	}
	
	size_t data_size = (size_t)input_struct.x * input_struct.y * sizeof(int16_t);
	CUDA_RETURN_IF_ERROR(cudaMemcpy(Session.d_input, input, data_size, cudaMemcpyHostToDevice));

	return true;
}

bool 
fully_sampled_beamform(const float* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	size_t total_count = dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;


	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	vol_config.voxel_counts = *(uint3*)&params.vol_counts;
	vol_config.total_voxels = (size_t)vol_config.voxel_counts.x * vol_config.voxel_counts.y * vol_config.voxel_counts.z;

	Session.volume_configuration = vol_config;
	
	Session.pitches.x = params.array_params.pitch[0];
	Session.pitches.y = params.array_params.pitch[1];

	Session.pulse_delay = params.pulse_delay;
	Session.decoded_dims = { params.decoded_dims[0] ,params.decoded_dims[1], params.decoded_dims[2] };

	cuComplex* d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_count * sizeof(cuComplex)));
	
	bool do_hilbert = true;
	if (do_hilbert)
	{
		hilbert::hilbert_transform(Session.d_decoded, Session.d_complex);
	}
	else
	{
		// Add a float between every value for the imaginary component
		CUDA_RETURN_IF_ERROR(cudaMemcpy2D(d_input, 2 * sizeof(float), input, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));
	}

	cuComplex* d_volume;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(float)));

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	
	uint delay_samples = round(params.pulse_delay * params.array_params.sample_freq);
	beamformer::beamform(d_volume, d_input, *(float3*)params.focus, samples_per_meter, params.f_number, delay_samples);

	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(float));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(cuComplex), cudaMemcpyDefault));


	return true;
}



bool readi_beamform_raw(const int16_t* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	uint2 raw_data_dims = *(uint2*)&(params.raw_dims);
	size_t total_count = (size_t)dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;
	size_t total_raw_count = (size_t)raw_data_dims.x * raw_data_dims.y;

	init_cuda_configuration(params.raw_dims, params.decoded_dims, params.channel_mapping);

	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	vol_config.voxel_counts = *(uint3*)&params.vol_counts;
	vol_config.total_voxels = (size_t)vol_config.voxel_counts.x * vol_config.voxel_counts.y * vol_config.voxel_counts.z;

	Session.volume_configuration = vol_config;

	Session.pitches.x = params.array_params.pitch[0];
	Session.pitches.y = params.array_params.pitch[1];

	Session.xdc_mins.x = params.array_params.xdc_mins[0];
	Session.xdc_mins.y = params.array_params.xdc_mins[1];

	Session.xdc_maxes.x = params.array_params.xdc_maxes[0];
	Session.xdc_maxes.y = params.array_params.xdc_maxes[1];

	Session.sequence = (TransmitModes)params.sequence;

	Session.pulse_delay = params.pulse_delay;

	Session.mixes_count = params.mixes_count;
	Session.mixes_offset = params.mixes_offset;
	memcpy(Session.mixes_rows, params.mixes_rows, sizeof(params.mixes_rows));

	i16* d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_raw_count * sizeof(i16)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input, total_raw_count * sizeof(i16), cudaMemcpyHostToDevice));

	// Recreate the fft plans for readi subgroup sized batches
	// TODO: Do this in a better way
	cufftDestroy(Session.forward_plan);
	cufftDestroy(Session.inverse_plan);
	cufftDestroy(Session.strided_plan);

	hilbert::plan_hilbert((int)Session.decoded_dims.x, (int)Session.decoded_dims.y * params.readi_group_size);

	i16_to_f::convert_data(d_input, Session.d_converted);

	Session.readi_group = params.readi_group_id;
	Session.readi_group_size = params.readi_group_size;

	hadamard::readi_decode(Session.d_converted, Session.d_decoded, params.readi_group_id, params.readi_group_size);

	bool do_hilbert = true;
	if (do_hilbert)
	{
		hilbert::hilbert_transform(Session.d_decoded, Session.d_complex);
	}
	else
	{
		CUDA_RETURN_IF_ERROR(cudaMemset(Session.d_complex, 0x00, total_count * sizeof(cuComplex)));
		CUDA_RETURN_IF_ERROR(cudaMemcpy2D(Session.d_complex, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));
	}

	cuComplex* d_volume;

	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(cuComplex)));

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	int delay_samples = round(params.pulse_delay * params.array_params.sample_freq);
	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(cuComplex));

	std::cout << "Starting beamform\n";

	beamformer::beamform(d_volume, Session.d_complex, *(float3*)params.focus, samples_per_meter, params.f_number, delay_samples);


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


	init_cuda_configuration(params.raw_dims, params.decoded_dims, params.channel_mapping);

	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[ 2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	vol_config.voxel_counts = *(uint3*) & params.vol_counts;
	vol_config.total_voxels = (size_t)vol_config.voxel_counts.x * vol_config.voxel_counts.y * vol_config.voxel_counts.z;
	Session.volume_configuration = vol_config;

	Session.pitches.x = params.array_params.pitch[0];
	Session.pitches.y = params.array_params.pitch[1];

	Session.xdc_mins.x = params.array_params.xdc_mins[0];
	Session.xdc_mins.y = params.array_params.xdc_mins[1];	
	
	Session.xdc_maxes.x = params.array_params.xdc_maxes[0];
	Session.xdc_maxes.y = params.array_params.xdc_maxes[1];

	Session.readi_group = params.readi_group_id;
	Session.readi_group_size = params.readi_group_size;

	Session.sequence = (TransmitModes)params.sequence;

	float *d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_raw_count * sizeof(float)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input, total_raw_count * sizeof(float), cudaMemcpyHostToDevice));

	//hadamard::readi_decode(d_input, Session.d_decoded, params.readi_group_id, params.readi_group_size);

	hadamard::readi_staggered_decode(d_input, Session.d_decoded, Session.d_hadamard);

	//CUDA_RETURN_IF_ERROR(cudaMemcpy(Session.d_decoded, d_input, total_raw_count * sizeof(float), cudaMemcpyDeviceToDevice));

	bool do_hilbert = true;
	if (do_hilbert)
	{
		hilbert::hilbert_transform(Session.d_decoded, Session.d_complex);
	}
	else
	{
		CUDA_RETURN_IF_ERROR(cudaMemset(Session.d_complex, 0x00, total_count * sizeof(cuComplex)));
		CUDA_RETURN_IF_ERROR(cudaMemcpy2D(Session.d_complex, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));
	}
	

	cuComplex* d_volume;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(cuComplex)));
	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(cuComplex));


	uint readi_group_count = dec_data_dims.z / Session.readi_group_size;
	Session.pitches.y = Session.pitches.y * (float)readi_group_count;

	Session.decoded_dims.z = Session.readi_group_size;

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	std::cout << "Starting beamform" << std::endl;
	uint delay_samples = round(params.pulse_delay * params.array_params.sample_freq);
	beamformer::beamform(d_volume, Session.d_complex, *(float3*)params.focus, samples_per_meter, params.f_number, delay_samples);

	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(cuComplex), cudaMemcpyDefault));

	cudaFree(d_input);
	cudaFree(d_volume);

	deinit_cuda_configuration();

	return true;
}