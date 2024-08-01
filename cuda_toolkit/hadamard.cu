#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hadamard.cuh"

__global__ void
hadamard::_kernels::generate_hadamard_kernel(float* hadamard, int prev_size, int final_size)
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

__host__
void hadamard::_host::print_array(float* out_array, uint size)
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

__global__ void
hadamard::_kernels::init_hadamard_matrix(float* matrix, int size)
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
hadamard::_host::generate_hadamard(uint size, float** dev_ptr)
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
		block_dim = { MAX_2D_BLOCK_DIM, MAX_2D_BLOCK_DIM, 1 };
	}

	grid_dim = { grid_length, grid_length, 1 };

	_kernels::init_hadamard_matrix << <grid_dim, block_dim >> > (*dev_ptr, size);


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

		_kernels::generate_hadamard_kernel << <grid_dim, block_dim >> > (*dev_ptr, i / 2, size);

		THROW_IF_ERROR(cudaGetLastError());
		THROW_IF_ERROR(cudaDeviceSynchronize());
	}

	return cudaSuccess;
}

__host__ bool 
hadamard::hadamard_decode(int sample_count, int channel_count, int tx_count, const float* input, const float* hadamard, float** output)
{

	uint hadamard_size = (uint)tx_count;
	float* d_hadamard = nullptr;

	size_t tx_size = sample_count * channel_count;

	cudaError_t status = _host::generate_hadamard(hadamard_size, &d_hadamard);

	size_t input_size = sample_count * channel_count * tx_count;
	float* d_input = nullptr;
	cudaMalloc((void**)&d_input, input_size * sizeof(float));

	*output = (float*)malloc(input_size * sizeof(float));
	float* d_output = nullptr;
	cudaMalloc((void**)&d_output, input_size * sizeof(float));


	cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
	int runs = 10;
	auto start = std::chrono::high_resolution_clock::now();
	auto end = start;
	std::chrono::duration<double> elapsed;

	for (int i = 0; i < runs; i++)
	{
		start = std::chrono::high_resolution_clock::now();
		CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tx_size, tx_count, tx_count, &alpha, input, tx_size, d_hadamard, tx_count, &beta, *output, tx_size));
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
		std::cout << "Multiply duration: " << elapsed.count() << " seconds" << std::endl;
	}

	
	cudaMemcpy(*output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_hadamard);

	return status == cudaSuccess;
}

__host__ cublasStatus_t
hadamard::_host::decode(const float* data, const float* hadamard, float* output, int3 data_dims)
{



	return CUBLAS_STATUS_SUCCESS;
}







