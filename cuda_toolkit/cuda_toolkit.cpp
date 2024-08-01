#include "defs.h"
#include "hilbert_transform.cuh"
#include "hadamard.cuh"

#include "cuda_toolkit.h"

result_t batch_hilbert_transform(int sample_count, int channel_count, const float* input, complex_f** output)
{ 
	bool success = hilbert_transform(sample_count, channel_count, input, reinterpret_cast<std::complex<float>**>(output));

	return success ? SUCCESS : FAILURE;
}

result_t hadamard_decode(int sample_count, int channel_count, int transmission_count, const float* input, float** output)
{
	bool success = hadamard::hadamard_decode(sample_count, channel_count, transmission_count, input, nullptr, output);

	return success ? SUCCESS : FAILURE;
}