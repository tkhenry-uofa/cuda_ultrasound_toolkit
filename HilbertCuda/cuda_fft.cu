#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include "cuda_fft.cuh"

/**
* Applies scaling to the spectrum to match the hilbert transform
* The first and middle values in each channel are left the same,
* values between them are doubled and values above are set to zero.
*
* The real to complex fft only calculates the first half of the spectrum,
* so the second half is still set to zero so we only need to update half double values
*/
__global__
void hilbert_scaling(cufftComplex* data, int fft_size)
{
	const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	const int pivot = floorf((float)fft_size / 2) + 1;
	const float scale_factor = 1.0f / (float)fft_size;
	const int location = thread_id % fft_size;
	cufftComplex* point = &(data[thread_id]);

	if (0 < location < pivot)
	{
		point->x *= 2 * scale_factor;
		point->y *= 2 * scale_factor;
	}
	else if (location > pivot)
	{
		point->x *= 0;
		point->y *= 0;
	}
	else
	{
		point->x *= scale_factor;
		point->y *= scale_factor;
	}
}

bool cuda_fft(const std::vector<float>& real_in, std::vector<std::complex<float>>** cpx_out, defs::RfDataDims dims)
{
	cufftResult_t fft_result;
	cudaError_t cuda_result;
	cufftHandle fwd_plan, inv_plan;
	cudaStream_t stream = nullptr;

	// cufft requires dims be in regular ints
	int fft_size = (int)dims.sample_count;
	int batch_count = (int)(dims.tx_count * dims.element_count);

	uint element_count = fft_size * batch_count;

	/*std::vector<float> test(16);

	for (float f = 0.0f; f < 16.0f; f++)
	{
		test.push_back(f);
	}*/

	float* d_input = nullptr;
	cufftComplex* d_output = nullptr;

	// Because the input is real the right half of the fft isn't calculated
	uint output_size = (fft_size / 2 + 1) * 2;

	// Make the output vector the size of the full signal so we can hilbert transform it
	*cpx_out = new std::vector<std::complex<float>>(element_count);

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

	// Stream setup for async
	/*cuda_result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	RETURN_IF_ERROR(cuda_result, "Failed to create stream.")
	fft_result = cufftSetStream(fwd_plan, stream);
	RETURN_IF_ERROR(fft_result, "Failed to set stream for forward fft.")
	fft_result = cufftSetStream(inv_plan, stream);	
	RETURN_IF_ERROR(fft_result, "Failed to set stream for forward fft.")*/

	// Malloc device arrays and copy input
	cuda_result = cudaMalloc((void**)&d_input, sizeof(float) * real_in.size());
	RETURN_IF_ERROR(cuda_result, "Failed to malloc input array.")
	cuda_result = cudaMalloc((void**)&d_output, sizeof(cufftComplex) * real_in.size());
	RETURN_IF_ERROR(cuda_result, "Failed to malloc output array.")

	cuda_result = cudaMemcpy(d_input, real_in.data(), sizeof(float) * real_in.size(), cudaMemcpyHostToDevice);
	RETURN_IF_ERROR(cuda_result, "Failed to memcpy input, error %d.")
	cuda_result = cudaMemset(d_output, 0x00, sizeof(cufftComplex) * real_in.size());
	RETURN_IF_ERROR(cuda_result, "failed to init output array to 0.")


	// Exec forward transform
	fft_result = cufftExecR2C(fwd_plan, d_input, d_output);
	RETURN_IF_ERROR(fft_result, "Failed to execute forward plan.")

	size_t grid_size = ceil(element_count / MAX_THREADS_PER_BLOCK);

	hilbert_scaling << <grid_size, MAX_THREADS_PER_BLOCK >> > (d_output, fft_size);

	// Exec reverse transform
	fft_result = cufftExecC2C(inv_plan, d_output, d_output, CUFFT_INVERSE);
	RETURN_IF_ERROR(fft_result, "Failed to execute inverse plan.")

	// Copy out results
	cuda_result = cudaMemcpy((*cpx_out)->data(), d_output, sizeof(std::complex<float>) * output_size, cudaMemcpyDeviceToHost);
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






