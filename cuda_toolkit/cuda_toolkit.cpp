

#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"

#include "cuda_toolkit.h"

CudaSession Session = { {0, 0}, {0, 0, 0}, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, false };

bool init_session(uint2 input_dims, uint3 decoded_dims)
{
	if (!Session.init)
	{
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


		size_t input_size = input_dims.x * input_dims.y * sizeof(i16);
		size_t output_size = decoded_dims.x * decoded_dims.y * decoded_dims.z * sizeof(float);

		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_input), input_size));
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_converted), output_size));
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_decoded), output_size));
		CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_complex), output_size * 2));

		float* d_hadamard = nullptr;
		CUDA_THROW_IF_ERROR(hadamard::generate_hadamard(decoded_dims.z, &(Session.d_hadamard)));

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
	cublasDestroy(Session.cublas_handle);
	cufftDestroy(Session.forward_plan);
	cufftDestroy(Session.inverse_plan);
	return true;
}

bool raw_data_to_cuda(const int16_t* input, uint32_t* input_dims, uint32_t* decoded_dims )
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		init_session(input_struct, decoded_struct);
	}
	
	size_t data_size = input_struct.x * input_struct.y * sizeof(int16_t);

	std::cout << "Data size: " << data_size << std::endl;
	CUDA_THROW_IF_ERROR(cudaMemcpy(Session.d_input, input, data_size, cudaMemcpyHostToDevice));

	return true;
}

result_t decode_and_hilbert(bool rx_rows, uint32_t output_buffer)
{

	if (!Session.init)
	{
		return FAILURE;
	}

	bool success;

	cudaGraphicsResource_t output_resource;
	std::cout << "Output buffer: " << output_buffer << std::endl;
	CUDA_THROW_IF_ERROR(cudaGraphicsGLRegisterBuffer(&output_resource, output_buffer, cudaGraphicsMapFlagsNone));
	CUDA_THROW_IF_ERROR(cudaGraphicsMapResources(1, &output_resource, 0));

	size_t num_bytes;
	cufftComplex* d_output = nullptr;
	CUDA_THROW_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));


	defs::RfDataDims data_dims = { Session.decoded_dims.x, Session.decoded_dims.y, Session.decoded_dims.z };
	success = i16_to_f::convert_data(Session.d_input, Session.d_converted, Session.input_dims, data_dims, rx_rows);
	success = hadamard::hadamard_decode(data_dims, Session.d_converted, Session.d_hadamard, Session.d_decoded);
	success = hilbert::hilbert_transform(Session.d_decoded, d_output);

	CUDA_THROW_IF_ERROR(cudaGraphicsUnmapResources(1, &output_resource, 0));
	CUDA_THROW_IF_ERROR(cudaGraphicsUnregisterResource(output_resource));

	return SUCCESS;
}


result_t test_convert_and_decode(const int16_t* input, uint32_t *input_dims, uint32_t *decoded_dims, bool rx_rows, float** output)
{
	uint2 input_dims_struct = { input_dims[0], input_dims[1] };
	defs::RfDataDims output_dims_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };

	size_t input_size = input_dims[0] * input_dims[1] * sizeof(i16);
	size_t output_size = decoded_dims[0] * decoded_dims[1] * decoded_dims[2] * sizeof(float);
	*output = (float*)malloc(output_size*2);

	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_input), input_size));
	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_converted), output_size));
	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_decoded), output_size));
	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(Session.d_complex), output_size * 2));

	float* d_hadamard = nullptr;
	CUDA_THROW_IF_ERROR(hadamard::generate_hadamard(output_dims_struct.tx_count, &d_hadamard));

	uint fft_channel_count = decoded_dims[1] * decoded_dims[2];
	uint sample_count = decoded_dims[0];

	hilbert::plan_hilbert(sample_count, fft_channel_count);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;

	int runs = 10;
	bool success;
	for (int i = 0; i < runs; i++)
	{
		start = std::chrono::high_resolution_clock::now();
		CUDA_THROW_IF_ERROR(cudaMemcpy(Session.d_input, input, input_size, cudaMemcpyHostToDevice));
		elapsed = std::chrono::high_resolution_clock::now() - start;
		std::cout << "Transfer duration: " << elapsed.count() << " Seconds." << std::endl;

		start = std::chrono::high_resolution_clock::now();
		success = i16_to_f::convert_data(Session.d_input, Session.d_converted, input_dims_struct, output_dims_struct, rx_rows);
		success = hadamard::hadamard_decode(output_dims_struct, Session.d_converted, d_hadamard, Session.d_decoded);
		elapsed = std::chrono::high_resolution_clock::now() - start;
		std::cout << "Decoding duration: " << elapsed.count() << " Seconds." << std::endl;

		start = std::chrono::high_resolution_clock::now();
		success = hilbert::hilbert_transform(Session.d_decoded, Session.d_complex);
		elapsed = std::chrono::high_resolution_clock::now() - start;
		std::cout << "Hilbert duration: " << elapsed.count() << " Seconds.\n" << std::endl;
	}

	CUDA_THROW_IF_ERROR(cudaMemcpy(*output, Session.d_complex, output_size*2, cudaMemcpyDeviceToHost));

	return success ? SUCCESS : FAILURE;
}