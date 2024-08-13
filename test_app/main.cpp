#include <iostream>
#include <string>
#include <chrono>

#include "../cuda_toolkit/cuda_toolkit.h"

#include "defs.h"
#include "parser/mat_parser.h"


bool test_decoding()
{
	std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\uforces_32_raw.mat)";
	std::string params_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\uforces_32_bp.mat)";
	std::string output_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\pipeline_output.mat)";

	defs::RfDataDims dims;
	std::vector<i16>* data_array = nullptr;

	parser::load_int16_array(input_file_path, &data_array, &dims);

	BeamformerParams params;

	parser::parse_bp_struct(params_file_path, &params);

	uint input_dims[2] = { dims.sample_count, dims.channel_count };
	size_t output_dims[3] = { 
		(size_t)params.decoded_dims[0], 
		(size_t)params.decoded_dims[1],
		(size_t)params.decoded_dims[2] };


	complex_f* intermediate = nullptr;
	complex_f* complex = nullptr;
	std::cout << "Processing" << std::endl;
	bool result = test_convert_and_decode(data_array->data(),params, &complex, &intermediate);

	result = parser::save_float_array(intermediate, output_dims, output_file_path, "intermediate", true);
	result = parser::save_float_array(complex, output_dims, output_file_path, "complex", true);


	free(complex);

	delete data_array;
	return true;
}

bool test_beamforming()
{
	std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\hero_acq.mat)";
	//std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\hercules_resolution_plane.mat)";
	std::string output_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\hero_apro.mat)";

	defs::RfDataDims dims;
	std::vector<float>* data_array = nullptr;

	parser::load_float_array(input_file_path, &data_array, &dims);

	BeamformerParams params;

	parser::parse_bp_struct(input_file_path, &params);

	uint input_dims[2] = { dims.sample_count, dims.channel_count };

	params.vol_mins[0] = -0.001f;
	params.vol_maxes[0] = 0.001f;

	params.vol_mins[1] = -0.03f;
	params.vol_maxes[1] = 0.03f;

	params.vol_mins[2] = 0.005f;
	params.vol_maxes[2] = 0.075f;

	params.lateral_resolution = 0.0002f;
	params.axial_resolution = 0.0001f;

	params.array_params.c = 1452;
	params.array_params.row_count = params.decoded_dims[1];
	params.array_params.col_count = params.decoded_dims[1];

	uint x_count, y_count, z_count;
	x_count = y_count = z_count = 0;
	for (float x = params.vol_mins[0]; x <= params.vol_maxes[0]; x += params.lateral_resolution) {
		x_count++;
	}
	for (float x = params.vol_mins[1]; x <= params.vol_maxes[1]; x += params.lateral_resolution) {
		y_count++;
	}
	for (float x = params.vol_mins[2]; x <= params.vol_maxes[2]; x += params.axial_resolution) {
		z_count++;
	}

	float* volume = nullptr;
	std::cout << "Processing" << std::endl;

	size_t vol_dims[3] = { x_count, y_count, z_count };
	
	bool result = hero_raw_to_beamfrom(data_array->data(), params, &volume);
	result = parser::save_float_array(volume, vol_dims, output_file_path, "volume", false);


	free(volume);

	delete data_array;

	return true;
}

int main()
{
	bool result = test_beamforming();
	return !result;
}