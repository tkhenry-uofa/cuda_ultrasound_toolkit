

#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"

#include "cuda_toolkit.h"

CudaSession Session = { false, {0, 0}, {0, 0, 0}, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL };

bool init_session(uint2 input_dims, uint3 decoded_dims, const uint channel_mapping[TOTAL_TOBE_CHANNELS])
{
	if (!Session.init)
	{

		i16_to_f::copy_channel_mapping(channel_mapping);

		Session.input_dims = input_dims;
		Session.decoded_dims = decoded_dims;

		cublasStatus_t cublas_result = cublasCreate(&(Session.cublas_handle));
		if (cublas_result != CUBLAS_STATUS_SUCCESS)
		{
			std::cerr << "Failed to create cublas handle" << std::endl;
			return false;
		}

		cufftResult_t cufft_result = cufftCreate(&(Session.forward_plan));
		if (cufft_result != CUFFT_SUCCESS)
		{
			std::cerr << "Failed to create forward cufft handle" << std::endl;
			return false;
		}

		cufft_result = cufftCreate(&(Session.inverse_plan));
		if (cufft_result != CUFFT_SUCCESS)
		{
			std::cerr << "Failed to create inverse cufft handle" << std::endl;
			return false;
		}


		size_t input_size = input_dims.x * input_dims.y;
		size_t decoded_size = decoded_dims.x * decoded_dims.y * decoded_dims.z;

		std::cout << "Allocing memory" << std::endl;
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_input), input_size * sizeof(i16)));
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_converted), decoded_size * sizeof(float)));
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_decoded), decoded_size * sizeof(float)));
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_complex), decoded_size * sizeof(cuComplex)));

		bool success = hadamard::generate_hadamard(decoded_dims.z, &(Session.d_hadamard));

		assert(success);

		uint fft_channel_count = decoded_dims.y * decoded_dims.z;
		uint sample_count = decoded_dims.x;

		hilbert::plan_hilbert(sample_count, fft_channel_count);

		Session.init = true;
	}
	
	return true;
}

bool startup()
{
	return true;
}

bool cleanup()
{
	cudaFree(Session.d_complex);
	cudaFree(Session.d_hadamard);
	cudaFree(Session.d_converted);
	cudaFree(Session.d_decoded);
	cudaFree(Session.d_input);
	cublasDestroy(Session.cublas_handle);
	cufftDestroy(Session.forward_plan);
	cufftDestroy(Session.inverse_plan);

	free(Session.channel_mapping);
	return true;
}

bool raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping )
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		init_session(input_struct, decoded_struct, channel_mapping);
	}
	
	size_t data_size = input_struct.x * input_struct.y * sizeof(int16_t);
	CUDA_THROW_IF_ERROR(cudaMemcpy(Session.d_input, input, data_size, cudaMemcpyHostToDevice));

	return true;
}

bool decode_and_hilbert(bool rx_cols, uint buffer_idx)
{
	
	auto start = std::chrono::high_resolution_clock::now();
	if (!Session.init)
	{
		std::cout << "Session not initialized" << std::endl;
		return false;
	}
	cudaGraphicsResource_t output_resource = Session.buffers[buffer_idx].cuda_resource;

	CUDA_THROW_IF_ERROR(cudaGraphicsMapResources(1, &output_resource));

	size_t total_count = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z;
	size_t output_size = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z * sizeof(cuComplex);

	uint sample_index = Session.decoded_dims.x;

	size_t num_bytes;
	cufftComplex *d_output = nullptr;
	CUDA_THROW_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	/*std::cout << "Bytes in pointer " << Session.buffers[buffer_idx].gl_buffer_id << ": " << num_bytes << std::endl;
	std::cout << "Decoder output size: " << output_size << std::endl;*/

	float sample;
	int16_t input_sample;

	i16_to_f::convert_data(Session.d_input, Session.d_converted, rx_cols);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);


	//CUDA_THROW_IF_ERROR(cudaMemcpy2D(d_output, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count,cudaMemcpyDefault));

	hilbert::hilbert_transform(Session.d_decoded, d_output);

	//Copy fist acq to all 32
	uint copy_length = Session.decoded_dims.x * Session.decoded_dims.y;
	for (int i = 1; i < Session.decoded_dims.z; i++)
	{
		cuComplex* row = d_output + i * copy_length;
		cudaMemcpy(row, d_output, copy_length * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
	}

	CUDA_THROW_IF_ERROR(cudaGraphicsUnmapResources(1, &output_resource));
	
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

bool register_cuda_buffers(uint* rf_data_ssbos, uint buffer_count)
{
	uint old_buffer_count = Session.buffer_count;

	if (old_buffer_count != 0)
	{
		for (uint i = 0; i < old_buffer_count; i++)
		{
			CUDA_THROW_IF_ERROR(cudaGraphicsUnregisterResource(Session.buffers[i].cuda_resource));
		}
		
		free(Session.buffers);
	}

	Session.buffers = (BufferMapping*)malloc(buffer_count * sizeof(BufferMapping));
	for (uint i = 0; i < buffer_count; i++)
	{
		std::cout << "Registering buffer : " << i << ", " << rf_data_ssbos[i] << std::endl;
		Session.buffers[i] = { NULL, rf_data_ssbos[i] };
		CUDA_THROW_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.buffers[i].cuda_resource), Session.buffers[i].gl_buffer_id, cudaGraphicsRegisterFlagsNone));
	}
	Session.buffer_count = buffer_count;

	return true;
}


bool test_convert_and_decode(const int16_t* input, const BeamformerParams params, complex_f** complex_out, float** intermediate)
{
	
	const uint2 input_dims = *(uint2*)&(params.raw_dims); // lol
	RfDataDims output_dims(params.decoded_dims);
	size_t input_size = input_dims.x * input_dims.y * sizeof(i16);
	size_t decoded_size = output_dims.sample_count * output_dims.channel_count * output_dims.tx_count * sizeof(float);
	size_t complex_size = decoded_size * 2;

	size_t total_count = output_dims.sample_count * output_dims.channel_count * output_dims.tx_count;

	*complex_out = (complex_f*)malloc(complex_size);
	*intermediate = (float*)malloc(32*32*4);
	float* d_intermediate;

	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(d_intermediate), decoded_size));

	raw_data_to_cuda(input, params.raw_dims, params.decoded_dims, params.channel_mapping);
	
	float sample;
	int16_t input_sample;



	i16_to_f::convert_data(Session.d_input, Session.d_converted, params.rx_cols);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	//hilbert::hilbert_transform(d_intermediate, Session.d_complex);
	CUDA_THROW_IF_ERROR(cudaMemcpy2D(Session.d_complex, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));

	CUDA_THROW_IF_ERROR(cudaMemcpy(*complex_out, Session.d_complex, complex_size, cudaMemcpyDeviceToHost));
	//CUDA_THROW_IF_ERROR(cudaMemcpy(*intermediate, Session.d_hadamard, 32*32*4, cudaMemcpyDeviceToHost));

	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	return true;
}