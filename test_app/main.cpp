#include <iostream>
#include <string>
#include <chrono>

#include "../cuda_toolkit/cuda_toolkit_testing.h"

#include "defs.h"
#include "parser/mat_parser.h"

#include "matlab_transfer.h"


PipelineParams convert_params(BeamformerParametersFull* full_bp)
{
	PipelineParams params;
	BeamformerParameters bp = full_bp->raw;

	params.focus[0] = 0.0f;
	params.focus[1] = 0.0f;
	params.focus[2] = bp.focal_depth;

	params.pulse_delay = bp.time_offset;

	params.decoded_dims[0] = bp.dec_data_dim.x;
	params.decoded_dims[1] = bp.dec_data_dim.y;
	params.decoded_dims[2] = bp.dec_data_dim.z;

	params.raw_dims[0] = bp.rf_raw_dim.x;
	params.raw_dims[1] = bp.rf_raw_dim.y;

	params.vol_mins[0] = bp.output_min_coordinate.x;
	params.vol_mins[1] = bp.output_min_coordinate.y;
	params.vol_mins[2] = bp.output_min_coordinate.z;

	params.vol_maxes[0] = bp.output_max_coordinate.x;
	params.vol_maxes[1] = bp.output_max_coordinate.y;
	params.vol_maxes[2] = bp.output_max_coordinate.z;

	params.vol_counts[0] = bp.output_points.x;
	params.vol_counts[1] = bp.output_points.y;
	params.vol_counts[2] = bp.output_points.z;

	params.vol_resolutions[0] = (params.vol_maxes[0] - params.vol_mins[0]) / params.vol_counts[0];
	params.vol_resolutions[1] = (params.vol_maxes[1] - params.vol_mins[1]) / params.vol_counts[1];
	params.vol_resolutions[2] = (params.vol_maxes[2] - params.vol_mins[2]) / params.vol_counts[2];

	for (int i = 0; i < 256; i++)
	{
		params.channel_mapping[i] = bp.channel_mapping[i];
	}

	params.channel_offset = bp.channel_offset;

	params.array_params.c = bp.speed_of_sound;
	params.array_params.center_freq = bp.center_frequency;
	params.array_params.sample_freq = bp.sampling_frequency;

	params.array_params.row_count = bp.dec_data_dim.y; // Assuming square arrays for now
	params.array_params.col_count = bp.dec_data_dim.y;

	params.array_params.xdc_mins[0] = bp.xdc_origin[0];
	params.array_params.xdc_mins[1] = bp.xdc_origin[1];


	// Origin is at (-x,-y), corner 1 is (+x,-y), corner 2 is (-x,+y)
	params.array_params.xdc_maxes[0] = bp.xdc_corner1[0];
	params.array_params.xdc_maxes[1] = bp.xdc_corner2[1];

	float width = params.array_params.xdc_maxes[0] - params.array_params.xdc_mins[0];
	params.array_params.pitch = width / params.array_params.row_count;

	params.readi_group_id = bp.readi_group_id;
	params.readi_group_size = bp.readi_group_size;

	return params;
}

bool beamform_from_fieldii()
{
	std::string data_root = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\vrs_data\)";
	std::string data_path = data_root + R"(field_ii)" + R"(\)";
	std::string input_file_path = data_path + R"(psf_plane.mat)";
	std::string output_file_path = data_root + R"(beamformed\psf_plane\)" + R"(8_stagger_apro.mat)";

	uint3 dims;
	std::vector<cuComplex>* data_array = nullptr;

	bool result = parser::load_complex_array(input_file_path, &data_array, &dims);
	if (!result) return false;

	PipelineParams params;
	result = parser::load_f2_tx_config(input_file_path, &params);
	if (!result) return false;

	params.decoded_dims[0] = dims.x;
	params.decoded_dims[1] = params.array_params.row_count;
	params.decoded_dims[2] = params.array_params.row_count;

	size_t vol_dims[3] = { 0,0,0 };
	{
		params.vol_mins[0] = -0.015f;
		params.vol_maxes[0] = 0.015f;

		params.vol_mins[1] = -0.015f;
		params.vol_maxes[1] = 0.015f;

		params.vol_mins[2] = 0.065f;
		params.vol_maxes[2] = 0.095f;

		float lateral_resolution = 0.0001f;
		float axial_resolution = 0.0001f;

		params.vol_resolutions[0] = lateral_resolution;
		params.vol_resolutions[1] = lateral_resolution;
		params.vol_resolutions[2] = axial_resolution;


		params.array_params.c = 1452;

		for (float x = params.vol_mins[0]; x <= params.vol_maxes[0]; x += lateral_resolution) {
			vol_dims[0]++;
		}
		for (float x = params.vol_mins[1]; x <= params.vol_maxes[1]; x += lateral_resolution) {
			vol_dims[1]++;
		}
		for (float x = params.vol_mins[2]; x <= params.vol_maxes[2]; x += axial_resolution) {
			vol_dims[2]++;
		}
	}

	cuComplex* volume = nullptr;
	std::cout << "Processing" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	result = fully_sampled_beamform((float*)data_array->data(), params, &volume);
	if (!result) return false;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Program duration: " << elapsed.count() << " seconds" << std::endl;

	result = parser::save_float_array(volume, vol_dims, output_file_path, "volume", false);
	if (!result) std::cout << "Failed to save volume" << std::endl;


	free(volume);
	delete data_array;

	return true;
}

bool test_beamforming()
{
	std::string data_root = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\vrs_data\)";
	std::string data_path = data_root + R"(Resolution_HERCULES-Diverging-TxColumn)" + R"(\)";
	std::string input_file_path = data_path + R"(49.mat)";
	std::string output_file_path = data_root + R"(beamformed\)" + R"(reso_mixes16.mat)";

	defs::RfDataDims dims;
	std::vector<i16>* data_array = nullptr;

	bool result = parser::load_int16_array(input_file_path, &data_array, &dims);
	if (!result) return false;

	PipelineParams params;
	result = parser::parse_bp_struct(input_file_path, &params);
	if (!result) return false;

	uint input_dims[2] = { dims.sample_count, dims.channel_count };

	params.vol_mins[0] = -0.015f;
	params.vol_maxes[0] = 0.015f;

	params.vol_mins[1] = -0.015f;
	params.vol_maxes[1] = 0.015f;

	params.vol_mins[2] = 0.04f;
	params.vol_maxes[2] = 0.06f;

	float lateral_resolution = 0.0001f;
	float axial_resolution = 0.0001f;

	params.vol_resolutions[0] = lateral_resolution;
	params.vol_resolutions[1] = lateral_resolution;
	params.vol_resolutions[2] = axial_resolution;

	params.array_params.c = 1454;
	params.array_params.row_count = params.decoded_dims[1];
	params.array_params.col_count = params.decoded_dims[1];

	params.array_params.pitch = (params.array_params.xdc_maxes[0] - params.array_params.xdc_mins[0]) / params.array_params.col_count;


	uint x_count, y_count, z_count;
	x_count = y_count = z_count = 0;
	for (float x = params.vol_mins[0]; x <= params.vol_maxes[0]; x += lateral_resolution) {
		x_count++;
	}
	for (float x = params.vol_mins[1]; x <= params.vol_maxes[1]; x += lateral_resolution) {
		y_count++;
	}
	for (float x = params.vol_mins[2]; x <= params.vol_maxes[2]; x += axial_resolution) {
		z_count++;
	}

	cuComplex* volume = nullptr;
	std::cout << "Processing" << std::endl;

	size_t vol_dims[3] = { x_count, y_count, z_count };

	auto start = std::chrono::high_resolution_clock::now();

	result = hero_raw_to_beamform(data_array->data(), params, &volume);
	if (!result) return false;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Program duration: " << elapsed.count() << " seconds" << std::endl;

	result = parser::save_float_array(volume, vol_dims, output_file_path, "volume", false);
	if (!result) return false;


	free(volume);
	delete data_array;

	return true;
}


bool readi_beamform_fii()
{
	BeamformerParametersFull* full_bp = nullptr;
	Handle input_pipe = nullptr;
	Handle output_pipe = nullptr;

	std::cout << "Main: Creating smem and input pipe." << std::endl;
	bool result = matlab_transfer::create_smem(&full_bp);

	if (!result)
	{
		std::cout << "Main: Failed to create smem." << std::endl;
		return false;
	}

	float* data_buffer = (float*)malloc(INPUT_MAX_BUFFER);
	uint bytes_read = 0;
	uint timeout = 2 * 60 * 60 * 1000; // 2 hours (for long simulations)

	result = matlab_transfer::create_input_pipe(&input_pipe);

	int max_beamforms = 1000; 
	// No state is carried over between iterations so this can handle multiple runs
	// All beamforming settings come from the state of the shared memory
	for (int g = 0; g < max_beamforms; g++)
	{
		std::cout << "Starting volume " << g + 1 << std::endl;
		if (!result)
		{
			std::cout << "Main: Failed to create input pipe." << std::endl;
			return false;
		}

		result = matlab_transfer::wait_for_data(input_pipe, data_buffer, &bytes_read, timeout);

		if (!result)
		{
			std::cout << "Error reading data from matlab." << std::endl;
			return false;
		}

		std::cout << "Restarting pipe" << std::endl;

		matlab_transfer::disconnect_pipe(input_pipe);
		matlab_transfer::close_pipe(input_pipe);
		input_pipe = nullptr;
		result = matlab_transfer::create_input_pipe(&input_pipe);

		std::cout << "Created input pipe, last error: " << matlab_transfer::last_error() << std::endl;

		if (!result)
		{
			std::cout << "Main: Failed to restart input pipe." << std::endl;
			return false;
		}

		// Now that we know matlab is up we can connect to the output pipe
		output_pipe = matlab_transfer::open_output_pipe(PIPE_OUTPUT_NAME);
		if (output_pipe == nullptr)
		{
			std::cout << "Error opening export pipe to matlab." << std::endl;
			return false;
		}

		// TODO: Unify structs and types so I don't have to deal with this 
		PipelineParams params = convert_params(full_bp);

		cuComplex* volume = nullptr;
		size_t output_size = full_bp->raw.output_points.x * full_bp->raw.output_points.y * full_bp->raw.output_points.z * sizeof(cuComplex);

		std::cout << "Starting pipeline " << g + 1 << std::endl;
		readi_beamform_fii(data_buffer, params, &volume);

		matlab_transfer::write_to_pipe(output_pipe, volume, output_size);
		matlab_transfer::close_pipe(output_pipe);

		std::cout << "Volume " << g + 1 << " done." << std::endl << std::endl;
	}

	free(data_buffer);

	return true;
}

bool readi_beamform()
{
	BeamformerParametersFull* full_bp = nullptr;
	Handle input_pipe = nullptr;
	Handle output_pipe = nullptr;

	std::cout << "Main: Creating smem and input pipe." << std::endl;
	bool result = matlab_transfer::create_smem(&full_bp);
	
	if (!result)
	{
		std::cout << "Main: Failed to create smem." << std::endl;
		return false;
	}

	i16* data_buffer = nullptr;
	uint bytes_read = 0;

	for (int g = 0; g < 16; g++)
	{

		std::cout << "Starting volume " << g + 1 << std::endl;
		uint timeout = 120000; // 2 mins

		result = matlab_transfer::create_input_pipe(&input_pipe);

		if (!result)
		{
			std::cout << "Main: Failed to create input pipe." << std::endl;
			return false;
		}

		result = matlab_transfer::wait_for_data(input_pipe, (void**)&data_buffer, &bytes_read, timeout);

		if (!result)
		{
			std::cout << "Error reading data from matlab." << std::endl;
			return false;
		}

		matlab_transfer::close_pipe(input_pipe);

		// Now that we know matlab is up we can connect to the output pipe
		output_pipe = matlab_transfer::open_output_pipe(PIPE_OUTPUT_NAME);
		if (output_pipe == nullptr)
		{
			std::cout << "Error opening export pipe to matlab." << std::endl;
			return false;
		}

		// TODO: Unify structs and types so I don't have to deal with this 
		PipelineParams params = convert_params(full_bp);

		cuComplex* volume = nullptr;
		size_t output_size = full_bp->raw.output_points.x * full_bp->raw.output_points.y * full_bp->raw.output_points.z * sizeof(cuComplex);

		std::cout << "Starting pipeline " << g + 1 << std::endl;
		readi_beamform_raw(data_buffer, params, &volume);

		free(data_buffer);

		matlab_transfer::write_to_pipe(output_pipe, volume, output_size);
		matlab_transfer::close_pipe(output_pipe);

		std::cout << "Volume " << g + 1 << " done." << std::endl;
	}

	
	return true;
}

int main()
{
	bool result = false;
	result = readi_beamform_fii();

	//result = readi_beamform();
	return !result;
}