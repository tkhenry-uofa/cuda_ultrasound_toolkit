#include <iostream>
#include <string>
#include <chrono>

#include "../cuda_toolkit/cuda_toolkit.h"

#include "defs.h"
#include "mat_parser.h"


bool test_decoding()
{
	std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\uforces_32_raw.mat)";
	std::string output_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\vrs_transfers\decoded.mat)";

	defs::RfDataDims dims;
	std::vector<i16>* data_array = nullptr;

	parser::load_int16_array(input_file_path, &data_array, &dims);

	uint input_dims[2] = { dims.sample_count, dims.channel_count };
	uint output_dims[3] = { 4352 , 128, 32 };
	size_t output_dims2[3] = { 4352 , 128, 32 };

	float* output = nullptr;
	bool result = test_convert_and_decode(data_array->data(), input_dims, output_dims, true, &output);
	cleanup();
	free(output);

	delete data_array;
	return true;
}

int main()
{
	bool result = test_decoding();
	return !result;
}