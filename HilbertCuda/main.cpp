#include <iostream>
#include <string>
#include <chrono>

#include "defs.h"
#include "mat_parser.h"

#include "cuda_fft.cuh"

int main()
{

	std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct_real.mat)";
	std::string output_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\fft_output.mat)";

	defs::RfDataDims dims;
	std::vector<float>* data_array = nullptr;

	parser::load_rf_data_array(input_file_path, &data_array, &dims);

	std::vector<std::complex<float>>* output;

	bool success = cuda_fft(*data_array, &output, dims);

	size_t output_dims[3] = { dims.sample_count, dims.element_count, dims.tx_count };
	success = parser::save_complex_data(output->data(), output_dims, output_file_path, "complex_data");

	delete data_array;
	delete output;
	return !success;
}