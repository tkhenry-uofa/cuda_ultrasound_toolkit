#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hilbert_transform.cuh"

__global__ void 
generate_hadamard_kernel(float* hadamard, int prev_size, int final_size)
{
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	bool top = row < prev_size;
	bool left = col < prev_size;

	// Index to get the value from the previous iteration
	int prev_row = top ? row : row - prev_size;
	int prev_col = left ? col : col - prev_size;
	
	if (!top && !left)
	{
		// If we are bottom right make a negative copy of top left
		hadamard[row * final_size + col] = -1.0 * hadamard[prev_row * final_size + prev_col];
	}
	else if (top != left)
	{
		// If we are top right or bottom left copy the top left value
		hadamard[row * final_size + col] = hadamard[prev_row * final_size + prev_col];
	}
}

__global__ void
init_hadamard_matrix(float* matrix, int size)
{
	int row = threadIdx.x * blockIdx.x;
	int col = threadIdx.y * blockIdx.y;

	if (row == 0 && col == 0)
	{
		matrix[0] = 1.0f;
	}
	else if (row < size && col < size)
	{
		matrix[row * size + col] = 0.0f;
	}
}

__host__ cudaError_t
generate_hadamard(uint size, float** dev_ptr)
{
	assert(ISPOWEROF2(size));

	if (*dev_ptr != nullptr)
	{
		THROW_IF_ERROR(cudaFree(*dev_ptr));
		*dev_ptr = nullptr;
	}

	size_t matrix_size = size * size * sizeof(float);
	THROW_IF_ERROR(cudaMalloc((void**)dev_ptr, matrix_size));

	uint grid_length;
	dim3 block_dim, grid_dim;

	if (size <= MAX_2D_BLOCK_DIM)
	{
		grid_length = 1;
		block_dim = { size, size, 1 };
	}
	else
	{
		grid_length = (uint)ceil((double)size / MAX_2D_BLOCK_DIM);
		block_dim = {MAX_2D_BLOCK_DIM, MAX_2D_BLOCK_DIM, 1 };
	}

	grid_dim = { grid_length, grid_length, 1 };

	init_hadamard_matrix<<<grid_dim, block_dim>>>(*dev_ptr, size);


	THROW_IF_ERROR(cudaGetLastError());
	THROW_IF_ERROR(cudaDeviceSynchronize());

	for (uint i = 2; i <= size; i *= 2)
	{
		if (i <= MAX_2D_BLOCK_DIM)
		{
			block_dim = { i, i, 1 };
			grid_dim = { 1, 1, 1 };
		}
		else
		{
			grid_length = (uint)ceil((double)i / MAX_2D_BLOCK_DIM);

			block_dim = { MAX_2D_BLOCK_DIM, MAX_2D_BLOCK_DIM, 1 };
			grid_dim = { grid_length, grid_length, 1 };
		}

		generate_hadamard_kernel<<<grid_dim, block_dim>>>(*dev_ptr, i / 2, size);

		THROW_IF_ERROR(cudaGetLastError());
		THROW_IF_ERROR(cudaDeviceSynchronize());
	}

	return cudaSuccess;
}

__host__
bool hilbert_transform(int sample_count, int channel_count, const float* input, std::complex<float>** output)
{
	cufftResult_t fft_result;
	cudaError_t cuda_result;
	cufftHandle fwd_plan, inv_plan;
	float* d_input = nullptr;
	cufftComplex* d_output = nullptr;

	uint data_size = sample_count * channel_count;
	*output = new std::complex<float>[data_size];

	fft_result = cufftCreate(&fwd_plan);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to create forward plan.\n")
	fft_result = cufftCreate(&inv_plan);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to create inverse plan.\n")

	// The FFT fails with CUDA_INTERNAL_ERROR if we don't estimate first, this only happens when FFT size isnt a power of 2
	// No idea what the cause it, it isn't in the docs anywhere.
	size_t work_size = 0;
	int dimensions[] = { sample_count };
	
	fft_result = cufftEstimateMany(1, &sample_count, dimensions, 1, sample_count, dimensions, 1, sample_count, CUFFT_R2C, channel_count, &work_size);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to estimate forward plan.")

	fft_result = cufftPlanMany(&fwd_plan, 1, &sample_count, dimensions, 1, sample_count, dimensions, 1, sample_count, CUFFT_R2C, channel_count);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to create forward plan.")

	fft_result = cufftPlanMany(&inv_plan, 1, &sample_count, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, channel_count);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to configure inverse plan.")

	// Malloc device arrays and copy input
	cuda_result = cudaMalloc((void**)&d_input, sizeof(float) * data_size);
	FFT_RETURN_IF_ERROR(cuda_result, "Failed to malloc input array.")
	cuda_result = cudaMalloc((void**)&d_output, sizeof(cufftComplex) * data_size);
	FFT_RETURN_IF_ERROR(cuda_result, "Failed to malloc output array.")

	cuda_result = cudaMemcpy(d_input, input, sizeof(float) * data_size, cudaMemcpyHostToDevice);
	FFT_RETURN_IF_ERROR(cuda_result, "Failed to copy input to device.")
	cuda_result = cudaMemset(d_output, 0x00, sizeof(cufftComplex) * data_size);
	FFT_RETURN_IF_ERROR(cuda_result, "Failed to init output array to 0.")

	// Exec forward transform
	auto start = std::chrono::high_resolution_clock::now();

	fft_result = cufftExecR2C(fwd_plan, d_input, d_output);
	FFT_RETURN_IF_ERROR(fft_result, "Failed to execute forward plan.")

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Forward duration: " << elapsed.count() << " seconds" << std::endl;

	// Exec reverse transform
	start = std::chrono::high_resolution_clock::now();
	fft_result = cufftExecC2C(inv_plan, d_output, d_output, CUFFT_INVERSE);

	elapsed = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Inverse duration: " << elapsed.count() << " seconds" << std::endl;
	FFT_RETURN_IF_ERROR(fft_result, "Failed to execute inverse plan.")

	cuda_result = cudaMemcpy((*output), d_output, sizeof(std::complex<float>) * data_size, cudaMemcpyDeviceToHost);
	FFT_RETURN_IF_ERROR(cuda_result, "Failed to copy result from host.")


	cufftDestroy(fwd_plan);
	cufftDestroy(inv_plan);
	cudaFree(d_input);
	cudaFree(d_output);

	return true;
}

__host__
bool hadamard_decode_cuda(int sample_count, int channel_count, int tx_count, const int* input, float** output)
{
	
	uint size = 64;
	size_t array_size = size * size * sizeof(float);

	float* cpu_array = (float*)malloc(array_size);
	float* device_array = nullptr;


	cudaError_t status = generate_hadamard(size, &device_array);

	
	status = cudaMemcpy(cpu_array, device_array, array_size, cudaMemcpyDeviceToHost);

	if (status == cudaSuccess)
	{
		print_array(cpu_array, size);
	}
	else
	{
		std::cout << "Failed out copy out data" << std::endl;
	}

	cudaFree(device_array);
	free(cpu_array);

	return status == cudaSuccess;
}

__host__
void print_array(float* out_array, uint size)
{
	std::cout << "Output" << std::endl;
	for (uint i = 0; i < size; i++)
	{
		for (uint j = 0; j < size; j++)
		{
			std::cout << out_array[i * size + j] << " ";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;
}





