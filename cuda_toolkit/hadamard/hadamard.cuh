#ifndef HADAMARD_CUH
#define HADAMARD_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>

#include "../defs.h"

namespace hadamard 
{
	__host__ bool 
	hadamard_decode(const float* d_input, float* d_output);

	__host__ bool
	generate_hadamard(uint size, float** dev_ptr);

	__host__ bool
	readi_staggered_decode(const float* d_input, float* d_output, float* d_hadamard);

	__host__ bool
	readi_decode(const float* d_input, float* d_output, uint group_number, uint group_size);

	__host__ bool
	c_readi_decode(const cuComplex* d_input, cuComplex* d_output, uint group_number, uint group_size);

	namespace _kernels
	{
		__global__ void
		generate_hadamard(float* hadamard, int prev_size, int final_size);

		__global__ void
		init_hadamard_matrix(float* matrix, int size);


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
		* Thread Values: [number_in_group, group_count]
		* Block Values: [sample, channel]
		*/
		__global__ void
		readi_staggered_decode_kernel(const float* d_input, float* d_output, const float* d_hadamard, uint readi_group, uint total_transmits);
	}

	namespace _host
	{
		__host__ void 
		print_array(float* out_array, uint size);
	}
}


#endif // !HADAMARD_CUH
