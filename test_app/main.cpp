#include <iostream>
#include <string>
#include <chrono>

#include "../cuda_toolkit/cuda_toolkit_testing.h"

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
	bool result = test_convert_and_decode(data_array->data(), params, &complex, &intermediate);

	result = parser::save_float_array(intermediate, output_dims, output_file_path, "intermediate", true);
	result = parser::save_float_array(complex, output_dims, output_file_path, "complex", true);


	free(complex);

	delete data_array;
	return true;
}

bool beamform_from_fieldii()
{
	std::string data_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\ql02\)";
	std::string input_file_path = data_path + R"(point_scan_200_-10.mat)";
	std::string output_file_path = data_path + R"(point_scan_200_-10_beamformed.mat)";

	uint3 dims;
	std::vector<cuComplex>* data_array = nullptr;

	bool result = parser::load_complex_array(input_file_path, &data_array, &dims);
	if (!result) return false;

	BeamformerParams params;
	result = parser::load_f2_tx_config(input_file_path, &params);
	if (!result) return false;

	params.decoded_dims[0] = dims.x;
	params.decoded_dims[1] = params.array_params.row_count;
	params.decoded_dims[2] = params.array_params.row_count;

	size_t vol_dims[3] = { 0,0,0 };
	{
		params.vol_mins[0] = -0.03f;
		params.vol_maxes[0] = 0.03f;

		params.vol_mins[1] = -0.02f;
		params.vol_maxes[1] = 0.02f;

		params.vol_mins[2] = 0.001f;
		params.vol_maxes[2] = 0.060f;

		params.lateral_resolution = 0.0002f;
		params.axial_resolution = 0.0002f;

		params.array_params.c = 1452;

		for (float x = params.vol_mins[0]; x <= params.vol_maxes[0]; x += params.lateral_resolution) {
			vol_dims[0]++;
		}
		for (float x = params.vol_mins[1]; x <= params.vol_maxes[1]; x += params.lateral_resolution) {
			vol_dims[1]++;
		}
		for (float x = params.vol_mins[2]; x <= params.vol_maxes[2]; x += params.axial_resolution) {
			vol_dims[2]++;
		}
	}

	float* volume = nullptr;
	std::cout << "Processing" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	result = fully_sampled_beamform((float*)data_array->data(), params, &volume);
	if (!result) return false;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Program duration: " << elapsed.count() << " seconds" << std::endl;

	result = parser::save_float_array(volume, vol_dims, output_file_path, "volume", false);
	if (!result) return false;


	free(volume);
	delete data_array;

	return true;

	return true;
}

bool test_beamforming()
{
	std::string data_root = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\vrs_data\)";
	std::string data_path = data_root + R"(cyst_herc_div_sin8)" + R"(\)";
	std::string input_file_path = data_path + R"(46.mat)";
	std::string output_file_path = data_root + R"(beamformed\)" + R"(cyst_herc_div_sin8_46.mat)";

	defs::RfDataDims dims;
	std::vector<i16>* data_array = nullptr;

	bool result = parser::load_int16_array(input_file_path, &data_array, &dims);
	if (!result) return false;

	BeamformerParams params;
	result = parser::parse_bp_struct(input_file_path, &params);
	if (!result) return false;

	uint input_dims[2] = { dims.sample_count, dims.channel_count };

	params.vol_mins[0] = -0.03f;
	params.vol_maxes[0] = 0.03f;

	params.vol_mins[1] = -0.03f;
	params.vol_maxes[1] = 0.03f;

	params.vol_mins[2] = 0.01f;
	params.vol_maxes[2] = 0.120f;

	params.lateral_resolution = 0.0003;;
	params.axial_resolution = 0.00015f;

	params.array_params.c = 1480;
	params.array_params.row_count = params.decoded_dims[1];
	params.array_params.col_count = params.decoded_dims[1];

	params.array_params.pitch = (params.array_params.xdc_maxes[0] - params.array_params.xdc_mins[0]) / params.array_params.col_count;

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

int main()
{
	bool result = test_beamforming();
	return !result;
}