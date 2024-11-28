#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"
#include "beamformer/beamformer.cuh"

#include "cuda_toolkit_testing.h"


bool generate_hero_location_array(ArrayParams params, float2** d_location)
{

	uint total_count = params.row_count * params.col_count;
	float2* cpu_array = (float2*)malloc(total_count * sizeof(float2));

	if (!cpu_array) return false; 
	
	float row_min = -1.0f * (params.row_count - 1) * params.pitch / 2;
	float col_min = -1.0f * (params.col_count - 1) * params.pitch / 2;

	float2 value;
	for (uint col = 0; col < params.col_count; col++)
	{
		for (uint row = 0; row < params.row_count; row++)
		{
			value = { row * params.pitch + row_min, col * params.pitch + col_min };
			cpu_array[col * params.row_count + row] = value;
		}
	}

	CUDA_RETURN_IF_ERROR(cudaMalloc(d_location, total_count * sizeof(float2)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*d_location, cpu_array, total_count * sizeof(float2), cudaMemcpyHostToDevice));

	free(cpu_array);
	return true;
}

bool raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const u16* channel_mapping )
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		_init_session(input_dims, decoded_dims, channel_mapping);
	}
	
	size_t data_size = input_struct.x * input_struct.y * sizeof(int16_t);
	CUDA_RETURN_IF_ERROR(cudaMemcpy(Session.d_input, input, data_size, cudaMemcpyHostToDevice));

	return true;
}

bool test_convert_and_decode(const int16_t* input, const PipelineParams params, complex_f** complex_out, complex_f** intermediate)
{
	const uint2 input_dims = *(uint2*)&(params.raw_dims); // lol
	const uint3 output_dims = *(uint3*)&(params.decoded_dims);
	size_t input_size = input_dims.x * input_dims.y * sizeof(i16);
	size_t decoded_size = output_dims.x * output_dims.y * output_dims.z * sizeof(float);
	size_t complex_size = decoded_size * 2;

	size_t total_count = output_dims.x * output_dims.y * output_dims.z;

	*complex_out = (complex_f*)malloc(complex_size);
	*intermediate = (complex_f*)malloc(complex_size);
	cuComplex* d_intermediate;

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(d_intermediate), complex_size));

	raw_data_to_cuda(input, params.raw_dims, params.decoded_dims, params.channel_mapping);

	Session.channel_offset = params.channel_offset;
	i16_to_f::convert_data(Session.d_input, Session.d_converted);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	hilbert::hilbert_transform2(Session.d_decoded, Session.d_complex, d_intermediate);

	//CUDA_RETURN_IF_ERROR(cudaMemcpy(*complex_out, Session.d_complex, complex_size, cudaMemcpyDeviceToHost));
	//CUDA_RETURN_IF_ERROR(cudaMemcpy(*intermediate, d_intermediate, complex_size, cudaMemcpyDeviceToHost));

	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

bool hero_raw_to_beamform(const int16_t* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	uint2 raw_data_dims = *(uint2*)&(params.raw_dims);
	size_t total_count = dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;
	size_t total_raw_count = raw_data_dims.x * raw_data_dims.y;


	init_cuda_configuration(params.raw_dims, params.decoded_dims, params.channel_mapping);

	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	beamformer::configure_volume(&vol_config);

	Session.volume_configuration = vol_config;

	Session.element_pitch = params.array_params.pitch;

	i16* d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_raw_count * sizeof(i16)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input, total_raw_count * sizeof(i16), cudaMemcpyHostToDevice));

	Session.channel_offset = params.channel_offset;
	i16_to_f::convert_data(d_input, Session.d_converted);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);
	hilbert::hilbert_transform2(Session.d_decoded, Session.d_complex, Session.d_complex);

	CUDA_RETURN_IF_ERROR(cudaMemset(Session.d_complex, 0x00, total_count * sizeof(cuComplex)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy2D(Session.d_complex, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));

	
	cuComplex* d_volume;

	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(float) * 2));

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	beamformer::beamform(d_volume, Session.d_complex, *(float3*)params.focus, samples_per_meter);

	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(float)*2);
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(float) * 2, cudaMemcpyDefault));

	cudaFree(d_input);

	return true;
}

bool 
fully_sampled_beamform(const float* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	size_t total_count = dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;

	cuComplex* input_c = (cuComplex*)input;
	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	beamformer::configure_volume(&vol_config);

	Session.volume_configuration = vol_config;
	Session.element_pitch = params.array_params.pitch;
	Session.pulse_delay = params.pulse_delay;
	Session.decoded_dims = { params.decoded_dims[0] ,params.decoded_dims[1], params.decoded_dims[2] };

	cuComplex* d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_count * sizeof(cuComplex)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input_c, total_count * sizeof(cuComplex), cudaMemcpyDefault));

	cuComplex* d_volume;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(float)));

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;
	beamformer::beamform(d_volume, d_input, *(float3*)params.focus, samples_per_meter);

	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(float));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(cuComplex), cudaMemcpyDefault));


	return true;
}



bool readi_beamform_raw(const int16_t* input, PipelineParams params, cuComplex** volume)
{
	uint3 dec_data_dims = *(uint3*)&(params.decoded_dims);
	uint2 raw_data_dims = *(uint2*)&(params.raw_dims);
	size_t total_count = dec_data_dims.x * dec_data_dims.y * dec_data_dims.z;
	size_t total_raw_count = raw_data_dims.x * raw_data_dims.y;


	init_cuda_configuration(params.raw_dims, params.decoded_dims, params.channel_mapping);

	VolumeConfiguration vol_config;
	vol_config.minimums = { params.vol_mins[0], params.vol_mins[1], params.vol_mins[2] };
	vol_config.maximums = { params.vol_maxes[0], params.vol_maxes[1], params.vol_maxes[2] };
	vol_config.axial_resolution = params.vol_resolutions[2];
	vol_config.lateral_resolution = params.vol_resolutions[0];

	beamformer::configure_volume(&vol_config);

	Session.volume_configuration = vol_config;

	Session.element_pitch = params.array_params.pitch;

	i16* d_input;
	CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, total_raw_count * sizeof(i16)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input, total_raw_count * sizeof(i16), cudaMemcpyHostToDevice));

	i16_to_f::convert_data(d_input, Session.d_converted);
	hadamard::readi_decode(Session.d_converted, Session.d_decoded, 1);
	hilbert::hilbert_transform(Session.d_decoded, Session.d_complex);

	//CUDA_RETURN_IF_ERROR(cudaMemset(Session.d_complex, 0x00, total_count * sizeof(cuComplex)));
	//CUDA_RETURN_IF_ERROR(cudaMemcpy2D(Session.d_complex, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));

	cuComplex* d_volume;

	CUDA_RETURN_IF_ERROR(cudaMalloc(&(d_volume), vol_config.total_voxels * sizeof(cuComplex)));

	float samples_per_meter = params.array_params.sample_freq / params.array_params.c;

	*volume = (cuComplex*)malloc(vol_config.total_voxels * sizeof(cuComplex));

	std::cout << "Starting beamform" << std::endl;
	beamformer::beamform(d_volume, Session.d_complex, *(float3*)params.focus, samples_per_meter);

	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*volume, d_volume, vol_config.total_voxels * sizeof(cuComplex), cudaMemcpyDefault));

	cudaFree(d_input);

	return true;
}