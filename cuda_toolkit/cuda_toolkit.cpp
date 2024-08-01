#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"

#include "cuda_toolkit.h"



result_t batch_hilbert_transform(int sample_count, int channel_count, const float* input, complex_f** output)
{ 
	bool success = hilbert_transform(sample_count, channel_count, input, reinterpret_cast<std::complex<float>**>(output));

	return success ? SUCCESS : FAILURE;
}

result_t hadamard_decode(int sample_count, int channel_count, int transmission_count, const float* input, float** output)
{
	//bool success = hadamard::hadamard_decode(sample_count, channel_count, transmission_count, input, nullptr, output);

	return SUCCESS;
}

result_t convert_and_decode(const int16_t* input, unsigned int input_dims[2], unsigned int decoded_dims[3], bool rx_rows, float** output)
{
	uint2 input_dims_struct = { input_dims[0], input_dims[1] };
	defs::RfDataDims output_dims_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };

	size_t input_size = input_dims[0] * input_dims[1] * sizeof(i16);
	size_t output_size = decoded_dims[0] * decoded_dims[1] * decoded_dims[2] * sizeof(float);

	i16* d_input;
	*output = (float*)malloc(output_size);
	float* d_converted;
	float* d_decoded;
	
	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&d_input, input_size));
	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&d_converted, output_size));
	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&d_decoded, output_size));

	float* d_hadamard = nullptr;
	CUDA_THROW_IF_ERROR(hadamard::generate_hadamard(output_dims_struct.tx_count, &d_hadamard));

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;

	start = std::chrono::high_resolution_clock::now();
	CUDA_THROW_IF_ERROR(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Transfer duration: " << elapsed.count() << " Seconds.\n" << std::endl;

	int runs = 10;
	bool success;
	

	for (int i = 0; i < runs; i++)
	{
		start = std::chrono::high_resolution_clock::now();
		success = i16_to_f::convert_data(d_input, d_converted, input_dims_struct, output_dims_struct, rx_rows);
		success = hadamard::hadamard_decode(output_dims_struct, d_converted, d_hadamard, d_decoded);
		elapsed = std::chrono::high_resolution_clock::now() - start;

		std::cout << "Decoding duration: " << elapsed.count() << " Seconds." << std::endl;
	}

	CUDA_THROW_IF_ERROR(cudaMemcpy(*output, d_decoded, output_size, cudaMemcpyDeviceToHost));

	cudaFree(d_input);
	cudaFree(d_converted);
	cudaFree(d_decoded);

	return success ? SUCCESS : FAILURE;
}