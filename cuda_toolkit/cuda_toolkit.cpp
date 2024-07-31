#include "defs.h"
#include "hilbert_transform.cuh"

#include "cuda_toolkit.h"

result_t batch_hilbert_transform(int sample_count, int channel_count, const float* input, complex_f** output)
{ 
	bool success = hilbert_transform(sample_count, channel_count, input, reinterpret_cast<std::complex<float>**>(output));

	return success ? SUCCESS : FAILURE;
}

result_t hadamard_decode()
{
	bool success = hadamard_decode_cuda(0, 0, 0, nullptr, nullptr);

	return success ? SUCCESS : FAILURE;
}