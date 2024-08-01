#include <iostream>
#include <string>
#include <chrono>

#include "cuda_toolkit.h"

#include "defs.h"
#include "mat_parser.h"

//#include "cuda_fft.cuh"


bool test_hilbert()
{
	std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct_real.mat)";
	std::string output_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\fft_output.mat)";

	defs::RfDataDims dims;
	std::vector<float>* data_array = nullptr;

	parser::load_float_array(input_file_path, &data_array, &dims);

	complex_f* output;

	//bool success = cuda_fft(data_array->data(), &output, dims);

	result_t hilbert_result = batch_hilbert_transform(dims.sample_count, dims.channel_count * dims.tx_count, data_array->data(), &output);

	if (hilbert_result != SUCCESS || output == NULL)
	{
		printf("ERROR\n");
	}

	size_t output_dims[3] = { dims.sample_count, dims.channel_count, dims.tx_count };
	bool success = parser::save_float_array(output, output_dims, output_file_path, "complex_data", true);

	delete data_array;
	delete[] output;
	return success;
}

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

	result_t result = convert_and_decode(data_array->data(), input_dims, output_dims, true, &output);

	//bool success = parser::save_float_array(output, output_dims2, output_file_path, "decoded", false);

	delete output;
	delete data_array;
	
	return SUCCESS;
}

int main()
{

	int result;

	//result = test_hilbert();

	result = test_decoding();


	return result;
}