#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hilbert_transform.cuh"


__host__
bool hilbert_transform(int sample_count, int channel_count, const float* input, std::complex<float>** output)
{
	cufftResult_t fft_result;
	cudaError_t cuda_result;
	cufftHandle fwd_plan, inv_plan;
	cudaStream_t stream = nullptr;
	float* d_input = nullptr;
	cufftComplex* d_output = nullptr;

	uint data_size = sample_count * channel_count;
	*output = new std::complex<float>[data_size];

	fft_result = cufftCreate(&fwd_plan);
	RETURN_IF_ERROR(fft_result, "Failed to create forward plan.\n")
	fft_result = cufftCreate(&inv_plan);
	RETURN_IF_ERROR(fft_result, "Failed to create inverse plan.\n")

	// The FFT fails with CUDA_INTERNAL_ERROR if we don't estimate first, this only happens when FFT size isnt a power of 2
	// No idea what the cause it, it isn't in the docs anywhere.
	size_t work_size = 0;
	int dimensions[] = { sample_count };
	
	fft_result = cufftEstimateMany(1, &sample_count, dimensions, 1, sample_count, dimensions, 1, sample_count, CUFFT_R2C, channel_count, &work_size);
	RETURN_IF_ERROR(fft_result, "Failed to estimate forward plan.")

	fft_result = cufftPlanMany(&fwd_plan, 1, &sample_count, dimensions, 1, sample_count, dimensions, 1, sample_count, CUFFT_R2C, channel_count);
	RETURN_IF_ERROR(fft_result, "Failed to create forward plan.")

	fft_result = cufftPlanMany(&inv_plan, 1, &sample_count, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, channel_count);
	RETURN_IF_ERROR(fft_result, "Failed to configure inverse plan.")

	// Malloc device arrays and copy input
	cuda_result = cudaMalloc((void**)&d_input, sizeof(float) * data_size);
	RETURN_IF_ERROR(cuda_result, "Failed to malloc input array.")
	cuda_result = cudaMalloc((void**)&d_output, sizeof(cufftComplex) * data_size);
	RETURN_IF_ERROR(cuda_result, "Failed to malloc output array.")

	cuda_result = cudaMemcpy(d_input, input, sizeof(float) * data_size, cudaMemcpyHostToDevice);
	RETURN_IF_ERROR(cuda_result, "Failed to copy input to device.")
	cuda_result = cudaMemset(d_output, 0x00, sizeof(cufftComplex) * data_size);
	RETURN_IF_ERROR(cuda_result, "Failed to init output array to 0.")

	// Exec forward transform
	auto start = std::chrono::high_resolution_clock::now();
	auto fft_start = start;

	fft_result = cufftExecR2C(fwd_plan, d_input, d_output);
	RETURN_IF_ERROR(fft_result, "Failed to execute forward plan.")

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Forward duration: " << elapsed.count() << " seconds" << std::endl;

	// Exec reverse transform
	start = std::chrono::high_resolution_clock::now();
	fft_result = cufftExecC2C(inv_plan, d_output, d_output, CUFFT_INVERSE);

	elapsed = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Inverse duration: " << elapsed.count() << " seconds" << std::endl;
	RETURN_IF_ERROR(fft_result, "Failed to execute inverse plan.")

	cuda_result = cudaMemcpy((*output), d_output, sizeof(std::complex<float>) * data_size, cudaMemcpyDeviceToHost);
	RETURN_IF_ERROR(cuda_result, "Failed to copy result from host.")


	cufftDestroy(fwd_plan);
	cufftDestroy(inv_plan);
	cudaFree(d_input);
	cudaFree(d_output);

	return true;
}






