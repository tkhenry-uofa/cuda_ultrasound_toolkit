#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include <iomanip>

#include <chrono>

#include "hadamard.cuh"

#define MAX_HADAMARD_SIZE 128

constexpr float hadamard_12_transpose[] = {
1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
1, -1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1,
1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1, -1,
1, -1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,
1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1,  1,
1,  1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1,
1,  1,  1,  1, -1,  1, -1, -1,  1, -1, -1, -1,
1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1, -1,
1, -1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,
1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1,  1,
1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1,
1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1,
};

constexpr float hadamard_20_transpose[] = {
1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,
1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,
1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,
1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,
1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,
1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,
1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,
1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,
1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,
1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,
1,   1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,
1,  -1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,
1,   1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,
1,   1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,
1,   1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,
1,   1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,
1,  -1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,
1,  -1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,
1,   1,  -1,  -1,   1,   1,  -1,  -1,  -1,  -1,   1,  -1,   1,  -1,   1,   1,   1,   1,  -1,  -1,
};


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
* Thread Value: [number_in_group]
* Block Values: [sample, channel]
*/
__global__ void
hadamard::_kernels::readi_staggered_decode_kernel(const float* d_input, float* d_output, const float* d_hadamard, uint readi_group, uint total_transmits)
{
	__shared__ float samples[MAX_HADAMARD_SIZE];
	int tx_in_group[MAX_HADAMARD_SIZE];
	int in_group_id = threadIdx.x;
	int group_size = blockDim.x;

	int group_count = total_transmits / group_size;

	int t = 0;
	for (int i = 0; i < total_transmits; i++)
	{
		if (i % group_count == readi_group)
		{
			tx_in_group[t] = i;
			t++;
		}
	}

	int thread_tx_id = tx_in_group[in_group_id];

	int sample_id = blockIdx.x;
	int channel_id = blockIdx.y;

	size_t io_offset = in_group_id * gridDim.x * gridDim.y + channel_id * gridDim.x + sample_id;

	samples[in_group_id] = d_input[io_offset];

	__syncthreads();

	float decoded_value = 0.0f;
	for (int i = 0; i < group_size; i++)
	{
		int transmit_id = tx_in_group[i];
		//decoded_value += samples[i] * d_hadamard[transmit_id + total_transmits * thread_tx_id];
		decoded_value += samples[i] * d_hadamard[i + total_transmits * in_group_id];
	}


	d_output[io_offset] = decoded_value;
}

__host__ bool
hadamard::readi_staggered_decode(const float* d_input, float* d_output, float* d_hadamard)
{
	dim3 grid_dim = { Session.decoded_dims.x, Session.decoded_dims.y, 1 };
	dim3 block_dim = { Session.readi_group_size, 1, 1 };

	_kernels::readi_staggered_decode_kernel << <grid_dim, block_dim >> > (d_input, d_output, d_hadamard, Session.readi_group, 128);

	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

__host__ bool
hadamard::generate_hadamard(uint requested_length, float** dev_ptr)
{
		
	uint base_length = 0;
	uint iterations = 0;

	size_t element_count = (size_t)requested_length * requested_length;
	float* cpu_hadamard = (float*)calloc(element_count, sizeof(float));
	if (!cpu_hadamard)
	{
		std::cerr << "Failed to init hadamard matrix of size " << requested_length << std::endl;
	}
		
	// Check if the requested length is valid and set up the base case.
	if (ISPOWEROF2(requested_length))
	{
		base_length = 1;
		iterations = (int)log2(requested_length);
		cpu_hadamard[0] = 1;
	}
	else if( requested_length % 12 == 0 && ISPOWEROF2(requested_length / 12))
	{
		base_length = 12;
		iterations = (int)log2(requested_length / 12);
		for (int i = 0; i < base_length; i++)
		{
			int source_offset = i * base_length;
			int dest_offset = i * requested_length;
			memcpy(cpu_hadamard + dest_offset, hadamard_12_transpose + source_offset, base_length * sizeof(float));
		}
	}
	else if (requested_length % 20 == 0 && ISPOWEROF2(requested_length / 20))
	{
		base_length = 20;
		iterations = (int)log2(requested_length / 20);
		for (int i = 0; i < base_length; i++)
		{
			int source_offset = i * base_length;
			int dest_offset = i * requested_length;
			memcpy(cpu_hadamard + dest_offset, hadamard_20_transpose + source_offset, base_length * sizeof(float));
		}
	}
	else
	{
		std::cerr << "Hadamard: Size '" << requested_length << "' not supported" << std::endl;
		std::cerr << "System only supports sizes of 2^n, 12 * 2^n or 20 * 2^n" << std::endl;
		return false;
	}
		
	// Recursively take the kroneker product of the matrix with H2 until we reach the desired size
	for (uint current_length = base_length; current_length < requested_length; current_length *= 2)
	{
		int right_half_offset = current_length; // Offset to top right half
		int bottom_half_offset = current_length * requested_length; // Offset to bottom left half
		for (int row_idx = 0; row_idx < current_length; row_idx++)
		{
			int row_offset = row_idx * requested_length;
			// Quadrants 1, 2, and 3 are copies of the base case, quadrant 4 is the negative of the base case
			for (int col_idx = 0; col_idx < current_length; col_idx++)
			{
				cpu_hadamard[col_idx + row_offset + right_half_offset] = cpu_hadamard[col_idx + row_offset];
				cpu_hadamard[col_idx + row_offset + bottom_half_offset] = cpu_hadamard[col_idx + row_offset];
				cpu_hadamard[col_idx + row_offset + bottom_half_offset + right_half_offset] = -1 * cpu_hadamard[col_idx + row_offset];
			}
		}
	}


	CUDA_NULL_FREE(*dev_ptr);
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)dev_ptr, element_count * sizeof(float)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(*dev_ptr, cpu_hadamard, element_count * sizeof(float), cudaMemcpyHostToDevice));
	free(cpu_hadamard);

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

	float alpha = 1/((float)dims.z); // Counters the N scaling of hadamard decoding 
	float beta = 0.0f;

	CUBLAS_RETURN_IF_ERR(cublasSgemm(
		Session.cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		tx_size, dims.z, dims.z,
		&alpha, d_input, tx_size,
		Session.d_hadamard, dims.z,
		&beta, d_output, tx_size));

	return true;
}

__host__ bool
hadamard::readi_decode(const float* d_input, float* d_output, uint group_number, uint group_size)
{

	uint row_count = Session.decoded_dims.z;

	float* d_hadamard_slice;
	cudaMalloc(&d_hadamard_slice, row_count * group_size * sizeof(float));

	uint hadamard_offset = group_number * group_size * row_count;

	uint3 dims = Session.decoded_dims;
	uint tx_size = dims.x * dims.y;

	cudaMemcpy(d_hadamard_slice, Session.d_hadamard + hadamard_offset, group_size * row_count * sizeof(float), cudaMemcpyDeviceToDevice);

	float alpha = 1 / ((float)dims.z); // Counters the N scaling of hadamard decoding 
	float beta = 0.0f;

	CUBLAS_RETURN_IF_ERR(cublasSgemm(Session.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, tx_size, dims.z, group_size, &alpha, d_input, tx_size, d_hadamard_slice, dims.z, &beta, d_output, tx_size));

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

	CUBLAS_RETURN_IF_ERR(cublasCgemm(Session.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, tx_size, dims.z, group_size, &alpha, d_input, tx_size, d_hadamard_slice, dims.z, &beta, d_output, tx_size));
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}



