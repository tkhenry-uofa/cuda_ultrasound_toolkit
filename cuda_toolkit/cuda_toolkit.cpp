#include "defs.h"
#include "hilbert/hilbert_transform.cuh"
#include "hadamard/hadamard.cuh"
#include "data_conversion/int16_to_float.cuh"
#include "beamformer/beamformer.cuh"

#include "cuda_toolkit.h"

CudaSession Session;

/*****************************************************************************************************************************
* Internal helpers
*****************************************************************************************************************************/

bool
_unregister_cuda_buffers()
{
	if (Session.raw_data_ssbo.cuda_resource != NULL)
	{
		CUDA_RETURN_IF_ERROR(cudaGraphicsUnregisterResource(Session.raw_data_ssbo.cuda_resource));
	}

	uint old_buffer_count = Session.rf_buffer_count;
	if (old_buffer_count != 0)
	{
		for (uint i = 0; i < old_buffer_count; i++)
		{
			CUDA_RETURN_IF_ERROR(cudaGraphicsUnregisterResource(Session.rf_data_ssbos[i].cuda_resource));
		}

		free(Session.rf_data_ssbos);
	}
	Session.rf_buffer_count = 0;

	return true;
}

bool
_cleanup_session()
{
	_unregister_cuda_buffers();
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

bool 
_init_session(const uint input_dims[2], const uint decoded_dims[3], const uint channel_mapping[256], bool rx_cols)
{
	if (!Session.init)
	{
		i16_to_f::copy_channel_mapping(channel_mapping);

		Session.input_dims.x = input_dims[0];
		Session.input_dims.y = input_dims[1];

		Session.decoded_dims.x = decoded_dims[0];
		Session.decoded_dims.y = decoded_dims[1];
		Session.decoded_dims.z = decoded_dims[2];


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

		size_t input_size = input_dims[0] * input_dims[1];
		size_t decoded_size = decoded_dims[0] * decoded_dims[1] * decoded_dims[2];

		//std::cout << "Allocing memory" << std::endl;
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_input), input_size * sizeof(i16)));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_converted), decoded_size * sizeof(float)));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_decoded), decoded_size * sizeof(float)));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_complex), decoded_size * sizeof(cuComplex)));

		bool success = hadamard::generate_hadamard(decoded_dims[2], &(Session.d_hadamard));

		assert(success);

		uint fft_channel_count = decoded_dims[1] * decoded_dims[2];
		uint sample_count = decoded_dims[0];

		hilbert::plan_hilbert(sample_count, fft_channel_count);

		Session.init = true;
	}
	
	return true;
}

/*****************************************************************************************************************************
* API FUNCTIONS
*****************************************************************************************************************************/

bool
init_cuda_configuration(const uint* input_dims, const uint* decoded_dims, const uint* channel_mapping, bool rx_cols)
{
	uint2 input_struct = { input_dims[0], input_dims[1] };
	uint3 decoded_struct = { decoded_dims[0], decoded_dims[1], decoded_dims[2] };
	if (!Session.init)
	{
		return _init_session(input_dims, decoded_dims, channel_mapping, rx_cols);
	}
	else
	{
		bool changed = input_struct.x != Session.input_dims.x || input_struct.y != Session.input_dims.y ||
			decoded_struct.x != Session.decoded_dims.x || decoded_struct.y != Session.decoded_dims.y || decoded_struct.z != Session.decoded_dims.z;

		if (changed)
		{
			_cleanup_session();
			return _init_session(input_dims, decoded_dims, channel_mapping, rx_cols);
		}
		return true;
	}
}

bool 
register_cuda_buffers(uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
	if (Session.rf_buffer_count != 0)
	{
		_unregister_cuda_buffers();
	}
	
	Session.raw_data_ssbo = { NULL, 0 };
	Session.raw_data_ssbo.gl_buffer_id = raw_data_ssbo;
	CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.raw_data_ssbo.cuda_resource), Session.raw_data_ssbo.gl_buffer_id, cudaGraphicsRegisterFlagsNone));

	Session.rf_data_ssbos = (BufferMapping*)malloc(rf_buffer_count * sizeof(BufferMapping));
	for (uint i = 0; i < rf_buffer_count; i++)
	{
		std::cout << "Registering buffer : " << i << ", " << rf_data_ssbos[i] << std::endl;
		Session.rf_data_ssbos[i] = { NULL, rf_data_ssbos[i] };
		CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.rf_data_ssbos[i].cuda_resource), Session.rf_data_ssbos[i].gl_buffer_id, cudaGraphicsRegisterFlagsNone));
	}
	Session.rf_buffer_count = rf_buffer_count;
	return true;
}

bool 
decode_and_hilbert(size_t input_offset, uint output_buffer)
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

	CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &output_resource));
	size_t total_count = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z;

	size_t num_bytes;
	i16* d_input = nullptr;
	cufftComplex *d_output = nullptr;
	CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &num_bytes, input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	d_input = d_input + input_offset / sizeof(i16);

	i16_to_f::convert_data(d_input, Session.d_converted, Session.rx_cols);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	// Skip hilbert transform and copy each decoded value to the real part of the output,
	CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, total_count * sizeof(cuComplex)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy2D(d_output, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count,cudaMemcpyDefault));
	
	//hilbert::hilbert_transform(Session.d_decoded, d_output);
	
	// Downsample copy
	//CUDA_THROW_IF_ERROR(cudaMemcpy2D(d_output, sizeof(cuComplex), Session.d_complex, 2*sizeof(cuComplex), sizeof(cuComplex), total_count/2, cudaMemcpyDefault));

	CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &output_resource));
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

