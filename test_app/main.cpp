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

	defs::BeamformerParams params;

	parser::parse_bp_struct(params_file_path, &params);

	uint input_dims[2] = { dims.sample_count, dims.channel_count };
	size_t output_dims[3] = { 
		(size_t)params.decoded_dims[0], 
		(size_t)params.decoded_dims[1],
		(size_t)params.decoded_dims[2] };


	complex_f* intermediate = nullptr;
	complex_f* complex = nullptr;
	std::cout << "Processing" << std::endl;
	bool result = test_convert_and_decode(data_array->data(), *(BeamformerParams*)&params, &complex, &intermediate);
	cleanup();

	result = parser::save_float_array(intermediate, output_dims, output_file_path, "intermediate", true);
	result = parser::save_float_array(complex, output_dims, output_file_path, "complex", true);


	free(complex);

	delete data_array;
	return true;
}

int main()
{
	bool result = test_decoding();
	return !result;
}