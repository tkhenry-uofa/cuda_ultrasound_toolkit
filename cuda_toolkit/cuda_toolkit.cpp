

#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"

#include "cuda_toolkit.h"

CudaSession Session = { false, false, {0, 0}, {0, 0, 0}, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, {NULL, 0}, NULL, 0, NULL};

bool init_session(uint2 input_dims, uint3 decoded_dims, const uint channel_mapping[TOTAL_TOBE_CHANNELS], bool rx_cols)
{
	if (!Session.init)
	{

		i16_to_f::copy_channel_mapping(channel_mapping);

		Session.input_dims = input_dims;
		Session.decoded_dims = decoded_dims;
		Session.rx_cols = rx_cols;

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

		//std::cout << "Allocing memory" << std::endl;
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

bool
unregister_cuda_buffers()
{
	if (Session.raw_data_ssbo.cuda_resource != NULL)
	{
		CUDA_THROW_IF_ERROR(cudaGraphicsUnregisterResource(Session.raw_data_ssbo.cuda_resource));
	}

	uint old_buffer_count = Session.rf_buffer_count;
	if (old_buffer_count != 0)
	{
		for (uint i = 0; i < old_buffer_count; i++)
		{
			CUDA_THROW_IF_ERROR(cudaGraphicsUnregisterResource(Session.rf_data_ssbos[i].cuda_resource));
		}

		free(Session.rf_data_ssbos);
	}
	Session.rf_buffer_count = 0;

	return true;
}

bool register_cuda_buffers(uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
	if (Session.rf_buffer_count != 0)
	{
		unregister_cuda_buffers();
	}
	
	Session.raw_data_ssbo = { NULL, 0 };
	Session.raw_data_ssbo.gl_buffer_id = raw_data_ssbo;
	CUDA_THROW_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.raw_data_ssbo.cuda_resource), Session.raw_data_ssbo.gl_buffer_id, cudaGraphicsRegisterFlagsNone));

	Session.rf_data_ssbos = (BufferMapping*)malloc(rf_buffer_count * sizeof(BufferMapping));
	for (uint i = 0; i < rf_buffer_count; i++)
	{
		std::cout << "Registering buffer : " << i << ", " << rf_data_ssbos[i] << std::endl;
		Session.rf_data_ssbos[i] = { NULL, rf_data_ssbos[i] };
		CUDA_THROW_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.rf_data_ssbos[i].cuda_resource), Session.rf_data_ssbos[i].gl_buffer_id, cudaGraphicsRegisterFlagsNone));
	}
	Session.rf_buffer_count = rf_buffer_count;
	return true;
}

bool startup()
{
	return true;
}

bool cleanup()
{
	unregister_cuda_buffers();
	cudaFree(Session.d_complex);
	cudaFree(Session.d_hadamard);
	cudaFree(Session.d_converted);
	cudaFree(Session.d_decoded);
	cudaFree(Session.d_input);
	cublasDestroy(Session.cublas_handle);
	cufftDestroy(Session.forward_plan);
	cufftDestroy(Session.inverse_plan);

	free(Session.channel_mapping);

	Session.init = false;
	return true;
}

bool raw_data_to_cuda(const int16_t* input, const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols )
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		init_session(input_struct, decoded_struct, channel_mapping, rx_cols);
	}
	
	size_t data_size = input_struct.x * input_struct.y * sizeof(int16_t);
	CUDA_THROW_IF_ERROR(cudaMemcpy(Session.d_input, input, data_size, cudaMemcpyHostToDevice));

	return true;
}

bool init_cuda_configuration(const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols)
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		return init_session(input_struct, decoded_struct, channel_mapping, rx_cols);
	}
	else
	{
		bool changed = input_struct.x != Session.input_dims.x || input_struct.y != Session.input_dims.y ||
			decoded_struct.x != Session.decoded_dims.x || decoded_struct.y != Session.decoded_dims.y || decoded_struct.z != Session.decoded_dims.z;

		if (changed)
		{
			deinit_cuda_configuration();
			return init_session(input_struct, decoded_struct, channel_mapping, rx_cols);
		}
		return true;
	}
}

bool deinit_cuda_configuration()
{
	cleanup();

	return true;
}

bool decode_and_hilbert(size_t input_offset, uint output_buffer)
{
	auto start = std::chrono::high_resolution_clock::now();
	if (!Session.init)
	{
		std::cout << "Session not initialized" << std::endl;
		return false;
	}

	cudaGraphicsResource_t input_resource = Session.raw_data_ssbo.cuda_resource;
	cudaGraphicsResource_t output_resource = Session.rf_data_ssbos[output_buffer].cuda_resource;

	if (!input_resource || !output_resource)
	{
		fprintf(stderr, "Open GL buffers not registered with cuda.");
		return false;
	}

	CUDA_THROW_IF_ERROR(cudaGraphicsMapResources(1, &input_resource));
	CUDA_THROW_IF_ERROR(cudaGraphicsMapResources(1, &output_resource));

	size_t total_count = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z;
	size_t output_size = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z * sizeof(cuComplex);

	size_t num_bytes;
	i16* d_input = nullptr;
	cufftComplex *d_output = nullptr;
	CUDA_THROW_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &num_bytes, input_resource));
	CUDA_THROW_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());
	
	d_input += input_offset / sizeof(i16);
	i16_to_f::convert_data(d_input, Session.d_converted, Session.rx_cols);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	//CUDA_THROW_IF_ERROR(cudaMemcpy2D(d_output, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count,cudaMemcpyDefault));
	
	hilbert::hilbert_transform(Session.d_decoded, d_output);
	
	// Downsample copy
	//CUDA_THROW_IF_ERROR(cudaMemcpy2D(d_output, sizeof(cuComplex), Session.d_complex, 2*sizeof(cuComplex), sizeof(cuComplex), total_count/2, cudaMemcpyDefault));

	CUDA_THROW_IF_ERROR(cudaGraphicsUnmapResources(1, &input_resource));
	CUDA_THROW_IF_ERROR(cudaGraphicsUnmapResources(1, &output_resource));
	
	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	// Advance the input pointer to the right buffer section
	size_t offset = 10000;
	cuComplex sample;
	cuComplex* d_sample = d_output;
	for (size_t i = 0; i < total_count; i += offset)
	{
		CUDA_THROW_IF_ERROR(cudaMemcpy(&sample, d_sample, sizeof(cuComplex), cudaMemcpyDefault));
		std::cout << "Offset " << i << " output Re: " << sample.x << " Im: " << sample.y << std::endl;
		d_sample += offset;
	}

	return true;
}

bool test_convert_and_decode(const int16_t* input, const BeamformerParams params, complex_f** complex_out, complex_f** intermediate)
{
	const uint2 input_dims = *(uint2*)&(params.raw_dims); // lol
	RfDataDims output_dims(params.decoded_dims);
	size_t input_size = input_dims.x * input_dims.y * sizeof(i16);
	size_t decoded_size = output_dims.sample_count * output_dims.channel_count * output_dims.tx_count * sizeof(float);
	size_t complex_size = decoded_size * 2;

	size_t total_count = output_dims.sample_count * output_dims.channel_count * output_dims.tx_count;

	*complex_out = (complex_f*)malloc(complex_size);
	*intermediate = (complex_f*)malloc(complex_size);
	cuComplex* d_intermediate;

	CUDA_THROW_IF_ERROR(cudaMalloc((void**)&(d_intermediate), complex_size));

	raw_data_to_cuda(input, params.raw_dims, params.decoded_dims, params.channel_mapping, params.rx_cols);

	i16_to_f::convert_data(Session.d_input, Session.d_converted, params.rx_cols);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	hilbert::hilbert_transform2(Session.d_decoded, Session.d_complex, d_intermediate);
//	CUDA_THROW_IF_ERROR(cudaMemcpy2D(Session.d_complex, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));

	CUDA_THROW_IF_ERROR(cudaMemcpy(*complex_out, Session.d_complex, complex_size, cudaMemcpyDeviceToHost));
	CUDA_THROW_IF_ERROR(cudaMemcpy(*intermediate, d_intermediate, complex_size, cudaMemcpyDeviceToHost));

	CUDA_THROW_IF_ERROR(cudaDeviceSynchronize());

	return true;
}