#include <iostream>
#include <string>

#include "defs.h"
#include "mat_parser.h"

#include "kernel.cuh"

int main()
{

	std::string file_path = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct_real.mat)";

	defs::RfDataDims dims;
	std::vector<float>* data_array = nullptr;

	parser::load_rf_data_array(file_path, &data_array, &dims);

	std::vector<std::complex<float>>* output;
	cuda_fft(*data_array, &output, dims);

	delete data_array;
	delete output;
	return 0;
}