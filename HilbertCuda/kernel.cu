#include <cufft.h>

#include "kernel.cuh"



bool cuda_fft(const std::vector<float>& real_in, std::vector<std::complex<float>>** cpx_out, defs::RfDataDims dims)
{

	cufftHandle fwd_plan, inv_plan;
	cudaStream_t stream = nullptr;

	uint fft_size = dims.sample_count;
	uint batch_count = dims.element_count * dims.tx_count;

	uint unit_count = fft_size * batch_count;

	float* d_input = nullptr;
	cufftComplex* d_output = nullptr;

	// Because the input is real the right half of the fft isn't calculated
	uint output_size = (fft_size / 2 + 1) * batch_count;

	// Make the output vector the size of the full signal so we can hilbert transform it
	*cpx_out = new std::vector<std::complex<float>>((fft_size) * batch_count);
	

	RETURN_IF_ERROR(cufftCreate(&fwd_plan), "Failed to create forward plan.\n")
	RETURN_IF_ERROR(cufftCreate(&inv_plan), "Failed to create inverse plan.\n")

	RETURN_IF_ERROR(cufftPlan1d(&fwd_plan, fft_size, CUFFT_R2C, batch_count), "Failed to configure forward plan.\n")
	RETURN_IF_ERROR(cufftPlan1d(&inv_plan, fft_size, CUFFT_C2R, batch_count), "Failed to configure inverse plan.\n")

//	RETURN_IF_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Failed to create stream.\n")

//	RETURN_IF_ERROR(cufftSetStream(fwd_plan, stream), "Failed to set forward stream.\n")
//	RETURN_IF_ERROR(cufftSetStream(inv_plan, stream), "Failed to set inverse stream.\n")

	RETURN_IF_ERROR(cudaMalloc((void**)&d_input, sizeof(float) * real_in.size()), "Failed to malloc input array")
	RETURN_IF_ERROR(cudaMalloc((void**)&d_output, sizeof(cufftComplex) * real_in.size()), "Failed to malloc output array")

	RETURN_IF_ERROR(cudaMemcpy(d_input, real_in.data(), sizeof(float) * real_in.size(), cudaMemcpyHostToDevice), "Failed to memcpy input.\n")


	cufftResult_t result = cufftExecR2C(fwd_plan, d_input, d_output);

	RETURN_IF_ERROR(result, "Forward fft failed.")


	RETURN_IF_ERROR(cudaMemcpy((*cpx_out)->data(), d_output, sizeof(std::complex<float>)* output_size, cudaMemcpyDeviceToHost), "Failed to copy forward result from host.\n")

	//RETURN_IF_ERROR(cudaStreamSynchronize(stream), "Failed to sync stream.\n")

	return true;
}






