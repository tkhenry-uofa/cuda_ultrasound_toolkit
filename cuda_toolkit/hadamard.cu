#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hadamard.cuh"

__global__ void
hadamard::kernels::generate_hadamard_kernel(float* hadamard, int prev_size, int final_size)
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
void hadamard::host::print_array(float* out_array, uint size)
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
hadamard::kernels::init_hadamard_matrix(float* matrix, int size)
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
hadamard::host::generate_hadamard(uint size, float** dev_ptr)
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

	kernels::init_hadamard_matrix << <grid_dim, block_dim >> > (*dev_ptr, size);


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

		kernels::generate_hadamard_kernel << <grid_dim, block_dim >> > (*dev_ptr, i / 2, size);

		THROW_IF_ERROR(cudaGetLastError());
		THROW_IF_ERROR(cudaDeviceSynchronize());
	}

	return cudaSuccess;
}

__host__
bool hadamard::hadamard_decode_cuda(int sample_count, int channel_count, int tx_count, const int* input, const float* hadamard, float** output)
{

	uint size = 64;
	size_t array_size = size * size * sizeof(float);

	float* cpu_array = (float*)malloc(array_size);
	float* device_array = nullptr;


	cudaError_t status = host::generate_hadamard(size, &device_array);


	status = cudaMemcpy(cpu_array, device_array, array_size, cudaMemcpyDeviceToHost);

	if (status == cudaSuccess)
	{
		host::print_array(cpu_array, size);
	}
	else
	{
		std::cout << "Failed out copy out data" << std::endl;
	}

	cudaFree(device_array);
	free(cpu_array);

	return status == cudaSuccess;
}






