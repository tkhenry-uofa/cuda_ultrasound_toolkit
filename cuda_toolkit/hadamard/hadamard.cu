#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <chrono>

#include "hadamard.cuh"

#define MAX_HADAMARD_SIZE 128

/**
* Takes in (S samples x C channels x T encoded transmits),
* Sorts each transmit into a group by (T % group_size = group_number)
* creating group_count sections.
*
* Each group is decoded using the rows and columns of the
* hadamard matrix matching the transmits in the set.
*
* Each (S x C x group_size) decoded set of data represents a
* (C x group_size) transducer with rectangular elements
*
* Each channels data should be a linear combination of the signal received
* by (group_size) ajdacent elements in the encoded direction
*
* (Just staggering the decodes means that after the first group its not a sum
* instead some elements are added and some are subtracted like in the hadamard matrix)
*/

/**
* Thread Values: [number_in_group, group_id]
* Block Values: [sample, channel]
*/
__global__ void
hadamard::_kernels::readi_staggered_decode_kernel(const float* d_input, float* d_output, const float* d_hadamard)
{

	__shared__ float samples[MAX_HADAMARD_SIZE];

	int group_id = threadIdx.y;
	int inGroup_id = threadIdx.x;
	int tx_id = group_id * blockDim.x + inGroup_id;

	int group_size = blockDim.x;
	int group_count = blockDim.y;

	int tx_count = blockDim.x * blockDim.y;
	
	int sample_id = blockIdx.x;
	int channel_id = blockIdx.y;

	size_t io_offset = tx_id * gridDim.x * gridDim.y + channel_id * gridDim.x + sample_id;

	samples[tx_id] = d_input[io_offset];

	int group_ids[MAX_HADAMARD_SIZE];
	int g = 0;
	// Get what other tx's are in this group
	for (int i = 0; i < tx_count; i++)
	{
		if (i % group_count == group_id)
		{
			group_ids[g] = i;
			g++;
		}
	}

	float hadamard_row[MAX_HADAMARD_SIZE];
	int hadamard_row_offset = tx_count * tx_id;
	for (int i = 0; i < group_size; i++)
	{

		hadamard_row[i] =  d_hadamard[i + hadamard_row_offset];
	}

	
	float decoded_value = 0.0f;
	for (int i = 0; i < group_size; i++)
	{
		int transmit_id = group_ids[i];
		decoded_value += samples[transmit_id] * hadamard_row[transmit_id];
	}

	d_output[io_offset] = decoded_value;
}

__host__ bool
hadamard::readi_staggered_decode(const float* d_input, float* d_output, float* d_hadamard, uint group_size, uint group_count)
{
	dim3 grid_dim = { Session.decoded_dims.x, Session.decoded_dims.y, 1 };
	dim3 block_dim = { group_size, group_count, 1 };

	_kernels::readi_staggered_decode_kernel << <grid_dim, block_dim >> > (d_input, d_output, d_hadamard);
	
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

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
hadamard::readi_decode(const float* d_input, float* d_output, uint group_number, uint group_size)
{

	uint row_count = Session.decoded_dims.z;

	float* d_hadamard_slice;
	cudaMalloc(&d_hadamard_slice, row_count * group_size * sizeof(float));

	int hadamard_offset = (group_number-1) * group_size * row_count; 

	uint3 dims = Session.decoded_dims;
	uint tx_size = dims.x * dims.y;

	cudaMemcpy(d_hadamard_slice, Session.d_hadamard + hadamard_offset, group_size * row_count * sizeof(float), cudaMemcpyDeviceToDevice);

	float alpha = 1.0f;
	//float alpha = 1/((float)dims.x/2);
	float beta = 0.0f;

	CUBLAS_THROW_IF_ERR(cublasSgemm(Session.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, tx_size, dims.z, group_size, &alpha, d_input, tx_size, d_hadamard_slice, dims.z, &beta, d_output, tx_size));

	return true;
}

__host__ bool
hadamard::c_readi_decode(const cuComplex* d_input, cuComplex* d_output, uint group_number, uint group_size)
{
	uint row_count = Session.decoded_dims.z;

	cuComplex* d_hadamard_slice;
	cudaMalloc(&d_hadamard_slice, row_count * group_size * sizeof(cuComplex));

	int hadamard_offset = (group_number - 1) * group_size * row_count;

	std::cout << "Hadamard offset " << hadamard_offset << std::endl;

	uint3 dims = Session.decoded_dims;
	uint tx_size = dims.x * dims.y;

	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_hadamard_slice, Session.d_c_hadamard + hadamard_offset, group_size * row_count * sizeof(cuComplex), cudaMemcpyDefault));
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	cuComplex alpha = { 1.0f, 0.0f };
	cuComplex beta = { 0.0f, 0.0f };

	CUBLAS_THROW_IF_ERR(cublasCgemm(Session.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, tx_size, dims.z, group_size, &alpha, d_input, tx_size, d_hadamard_slice, dims.z, &beta, d_output, tx_size));
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

