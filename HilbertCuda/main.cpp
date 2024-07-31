#include <iostream>
#include <string>
#include <chrono>

#include "cuda_toolkit.h"

#include "defs.h"
#include "mat_parser.h"

//#include "cuda_fft.cuh"

int main()
{

	std::string input_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct_real.mat)";
	std::string output_file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\fft_output.mat)";

	defs::RfDataDims dims;
	std::vector<float>* data_array = nullptr;

	parser::load_rf_data_array(input_file_path, &data_array, &dims);

	complex_f* output;

	//bool success = cuda_fft(data_array->data(), &output, dims);

	result_t hilbert_result = batch_hilbert_transform(dims.sample_count, dims.element_count * dims.tx_count, data_array->data(), &output);

	if (hilbert_result != SUCCESS || output == NULL)
	{
		printf("ERROR\n");
	}

	size_t output_dims[3] = { dims.sample_count, dims.element_count, dims.tx_count };
	bool success = parser::save_complex_data(reinterpret_cast<defs::ComplexF*>(output), output_dims, output_file_path, "complex_data");

	delete data_array;
	delete[] output;
	return !success;
}