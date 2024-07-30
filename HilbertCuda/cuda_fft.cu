#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "cuda_fft.cuh"

/**
* 1D kernel
* Applies scaling to the spectrum to match the hilbert transform
* 
* The transform function h(data) is defined as:
* 
* data[x],   x = 0, N/2+1
* data[x]*2, 0 < x < N/2+1
* 0,		 N/2+1 < x < N
* 
* Real to complex ffts in CUFFT do not calculate the N/2+2 and higher terms
* as the FFT is symmetric, so values above N/2+1 are expected to already be 0.
*/
__global__
void hilbert_scaling(cuFloatComplex* data, int N, int pivot)
{
	const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	cuFloatComplex value = data[thread_id];

	value = cuCmulf(value, { 2.0f, 2.0f });

	// thread_id % fft_size gives us the location within this channel's data
	if (0 < thread_id % N < pivot)
	{
		data[thread_id] = value;
	}
}

__host__
bool cuda_fft(const float* input, defs::ComplexF** output, defs::RfDataDims dims)
{
	cufftResult_t fft_result;
	cudaError_t cuda_result;
	cufftHandle fwd_plan, inv_plan;
	cudaStream_t stream = nullptr;
	float* d_input = nullptr;
	cufftComplex* d_output = nullptr;

	// cufft requires dims be in regular ints
	int fft_size = (int)dims.sample_count;
	int batch_count = (int)(dims.tx_count * dims.element_count);
	uint data_size = fft_size * batch_count;

	*output = new defs::ComplexF[data_size];

	fft_result = cufftCreate(&fwd_plan);
	RETURN_IF_ERROR(fft_result, "Failed to create forward plan.\n")
	fft_result = cufftCreate(&inv_plan);
	RETURN_IF_ERROR(fft_result, "Failed to create inverse plan.\n")

	// The FFT fails with CUDA_INTERNAL_ERROR if we don't estimate first, this only happens when FFT size isnt a power of 2
	// No idea what the cause it, it isn't in the docs anywhere.
	size_t work_size = 0;
	int dimensions[] = { fft_size };
	
	fft_result = cufftEstimateMany(1, &fft_size, dimensions, 1, fft_size, dimensions, 1, fft_size, CUFFT_R2C, batch_count, &work_size);
	RETURN_IF_ERROR(fft_result, "Failed to estimate forward plan.")

	fft_result = cufftPlanMany(&fwd_plan, 1, &fft_size, dimensions, 1, fft_size, dimensions, 1, fft_size, CUFFT_R2C, batch_count);
	RETURN_IF_ERROR(fft_result, "Failed to create forward plan.")

	fft_result = cufftPlanMany(&inv_plan, 1, &fft_size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch_count);
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

	/*uint grid_size = (uint)ceil(data_size / THREADS_PER_BLOCK);
	start = std::chrono::high_resolution_clock::now();
	hilbert_scaling << <grid_size, THREADS_PER_BLOCK >> > ((cuFloatComplex*)d_output, fft_size, (int)floorf((float)fft_size/2) + 1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;*/

	// Exec reverse transform
	start = std::chrono::high_resolution_clock::now();
	fft_result = cufftExecC2C(inv_plan, d_output, d_output, CUFFT_INVERSE);

	elapsed = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Inverse duration: " << elapsed.count() << " seconds" << std::endl;
	RETURN_IF_ERROR(fft_result, "Failed to execute inverse plan.")

	cuda_result = cudaMemcpy((*output), d_output, sizeof(std::complex<float>) * data_size, cudaMemcpyDeviceToHost);
	RETURN_IF_ERROR(cuda_result, "Failed to copy result from host.")

	// Cleanup
	/*cuda_result = cudaStreamSynchronize(stream);
	RETURN_IF_ERROR(cuda_result, "Failed to sync stream.")*/

	cufftDestroy(fwd_plan);
	cufftDestroy(inv_plan);
	cudaFree(d_input);
	cudaFree(d_output);

	return true;
}






