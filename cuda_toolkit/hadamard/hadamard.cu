#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hadamard.cuh"

__global__ void
hadamard::_kernels::generate_hadamard(float* hadamard, int prev_size, int final_size)
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

__host__ void 
hadamard::_host::print_array(float* out_array, uint size)
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

__host__ bool
hadamard::generate_hadamard(uint size, float** dev_ptr)
{
	if (!(ISPOWEROF2(size)))
	{
		return false;
	}

	CUDA_NULL_FREE(*dev_ptr);

	size_t matrix_size = size * size * sizeof(float);
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)dev_ptr, matrix_size));

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


	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

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

		_kernels::generate_hadamard << <grid_dim, block_dim >> > (*dev_ptr, i / 2, size);

		CUDA_RETURN_IF_ERROR(cudaGetLastError());
		CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	}

	return true;
}

__host__ bool 
hadamard::hadamard_decode(const float* d_input, float* d_output)
{

	if (!Session.hadamard_generated)
	{
		std::cerr << "Hadamard: Attempted to decode without a valid hadamard matrix" << std::endl;
		return false;
	}

	uint3 dims = Session.decoded_dims;
	uint tx_size = dims.x * dims.y;

	float alpha = 1.0f;
	//float alpha = 1/((float)dims.x/2);
	float beta = 0.0f;

	CUBLAS_THROW_IF_ERR(cublasSgemm(Session.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, tx_size, dims.z, dims.z, &alpha, d_input, tx_size, Session.d_hadamard, dims.z, &beta, d_output, tx_size));

	return true;
}

__host__ bool
hadamard::readi_decode(const float* d_input, float* d_output, int group_number)
{

	int row_count = 128;
	int group_size = 8;
	int group_count = 16;

	float* hadamard_slice;
	cudaMalloc(&hadamard_slice, row_count * group_size * sizeof(float));

	int hadamard_offset = group_number * group_size; 

	cudaMemcpy(hadamard_slice, d_input + hadamard_offset, group_size * sizeof(float), cudaMemcpyDeviceToDevice);

	uint3 dims = Session.decoded_dims;
	uint tx_size = dims.x * dims.y;

	float alpha = 1.0f;
	//float alpha = 1/((float)dims.x/2);
	float beta = 0.0f;

	CUBLAS_THROW_IF_ERR(cublasSgemm(Session.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, tx_size, row_count, group_size, &alpha, d_input, tx_size, Session.d_hadamard, dims.z, &beta, d_output, tx_size));


	return true;
}








