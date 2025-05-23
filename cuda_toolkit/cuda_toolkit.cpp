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

static bool
unregister_cuda_buffers()
{
	if (Session.raw_data_ssbo.cuda_resource != nullptr)
	{
		CUDA_RETURN_IF_ERROR(cudaGraphicsUnregisterResource(Session.raw_data_ssbo.cuda_resource));
		Session.raw_data_ssbo.cuda_resource = nullptr;
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

static bool
cleanup_session()
{
	CUDA_NULL_FREE(Session.d_complex);
	CUDA_NULL_FREE(Session.d_hadamard);
	CUDA_NULL_FREE(Session.d_c_hadamard);
	CUDA_NULL_FREE(Session.d_converted);
	CUDA_NULL_FREE(Session.d_decoded);
	CUDA_NULL_FREE(Session.d_input);
	CUDA_NULL_FREE(Session.d_match_filter);
	cublasDestroy(Session.cublas_handle);
	cufftDestroy(Session.forward_plan);
	cufftDestroy(Session.inverse_plan);
	cufftDestroy(Session.strided_plan);

	if (Session.channel_mapping)
	{
		free(Session.channel_mapping);
		Session.channel_mapping = nullptr;
	}
	Session.init = false;
	return true;
}

static bool 
init_session(const uint input_dims[2], const uint decoded_dims[3])
{
	if (!Session.init)
	{

		Session.input_dims.x = input_dims[0];
		Session.input_dims.y = input_dims[1];

		Session.decoded_dims.x = decoded_dims[0];
		Session.decoded_dims.y = decoded_dims[1];
		Session.decoded_dims.z = decoded_dims[2];


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

		size_t input_size = (size_t)input_dims[0] * input_dims[1];
		size_t decoded_size = (size_t)decoded_dims[0] * decoded_dims[1] * decoded_dims[2];

		//std::cout << "Allocing memory" << std::endl;
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_input), input_size * sizeof(i16)));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_converted), decoded_size * sizeof(float)));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_decoded), decoded_size * sizeof(float)));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&(Session.d_complex), decoded_size * sizeof(cuComplex)));

		bool success = hadamard::generate_hadamard(Session.decoded_dims.z, &(Session.d_hadamard));

		Session.hadamard_generated = success;

		uint fft_channel_count = decoded_dims[1] * decoded_dims[2];
		uint sample_count = decoded_dims[0];

		hilbert::plan_hilbert((int)sample_count, (int)fft_channel_count);

		Session.init = true;
	}
	
	return true;
}

/*****************************************************************************************************************************
* API FUNCTIONS
*****************************************************************************************************************************/

bool
init_cuda_configuration(const uint* input_dims, const uint* decoded_dims)
{
	if (!Session.init)
	{
		return init_session(input_dims, decoded_dims);
	}
	else
	{
		bool changed = input_dims[0] != Session.input_dims.x || input_dims[1] != Session.input_dims.y ||
			decoded_dims[0] != Session.decoded_dims.x || decoded_dims[1] != Session.decoded_dims.y ||
			decoded_dims[2] != Session.decoded_dims.z;

		if (changed)
		{
			std::cout << "configuration changed" << std::endl;
			cleanup_session();
			return init_session(input_dims, decoded_dims);
		}
		return true;
	}
}

bool 
cuda_set_channel_mapping(const i16 channel_mapping[MAX_CHANNEL_COUNT])
{
	if(!Session.channel_mapping)
	{
		Session.channel_mapping = (i16*)malloc(MAX_CHANNEL_COUNT * sizeof(i16));
	}

	memcpy(Session.channel_mapping, channel_mapping, MAX_CHANNEL_COUNT * sizeof(i16));
	return i16_to_f::copy_channel_mapping(channel_mapping);
}

bool
cuda_set_match_filter(const float* match_filter, uint length)
{
	if (Session.d_match_filter)
	{
		CUDA_NULL_FREE(Session.d_match_filter);
	}

	if (length == 0)
	{
		Session.match_filter_length = 0;
		return true;
	}
	else
	{
		Session.match_filter_length = length;
		hilbert::setup_filter(Session.decoded_dims.x ,length, match_filter);
	}
	return true;
}

bool 
register_cuda_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
	if (Session.rf_buffer_count != 0)
	{
		unregister_cuda_buffers();
	}
	
	Session.raw_data_ssbo = { nullptr, 0 };
	Session.raw_data_ssbo.gl_buffer_id = raw_data_ssbo;
	CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.raw_data_ssbo.cuda_resource), Session.raw_data_ssbo.gl_buffer_id, cudaGraphicsRegisterFlagsNone));

	Session.rf_data_ssbos = (BufferMapping*)malloc(rf_buffer_count * sizeof(BufferMapping));
	for (uint i = 0; i < rf_buffer_count; i++)
	{
		Session.rf_data_ssbos[i] = { NULL, rf_data_ssbos[i] };
		CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(Session.rf_data_ssbos[i].cuda_resource), Session.rf_data_ssbos[i].gl_buffer_id, cudaGraphicsRegisterFlagsNone));
	}
	Session.rf_buffer_count = rf_buffer_count;
	return true;
}

bool
cuda_decode(size_t input_offset, uint output_buffer_idx)
{
	if (!Session.init)
	{
		std::cout << "Session not initialized" << std::endl;
		return false;
	}

	cudaGraphicsResource_t input_resource = Session.raw_data_ssbo.cuda_resource;
	cudaGraphicsResource_t output_resource = Session.rf_data_ssbos[output_buffer_idx].cuda_resource;
	if (!input_resource || !output_resource)
	{
		std::cout << "Open GL buffers not registered with cuda.\n";
		return false;
	}

	CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &output_resource));
	const size_t total_count = (size_t)Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z;

	size_t num_bytes;
	i16* d_input = nullptr;
	cufftComplex* d_output = nullptr;
	CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &num_bytes, input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	d_input = d_input + input_offset / sizeof(i16);

	bool result = i16_to_f::convert_data(d_input, Session.d_converted);
	result = hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	// Insert 0s between each value for their imaginary components
	CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, total_count * sizeof(cuComplex)));
	CUDA_RETURN_IF_ERROR(cudaMemcpy2D(d_output, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));

	CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &output_resource));
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return result;
}

bool
cuda_hilbert(uint input_buffer_idx, uint output_buffer_idx)
{
	if (!Session.init)
	{
		std::cout << "Session not initialized" << std::endl;
		return false;
	}

	cudaGraphicsResource_t input_resource = Session.rf_data_ssbos[input_buffer_idx].cuda_resource;
	cudaGraphicsResource_t output_resource = Session.rf_data_ssbos[output_buffer_idx].cuda_resource;
	if (!input_resource || !output_resource)
	{
		fprintf(stderr, "Open GL buffers not registered with cuda.");
		return false;
	}

	CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &output_resource));

	size_t num_bytes;
	cuComplex* d_input = nullptr;
	cuComplex* d_output = nullptr;
	CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &num_bytes, input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));

	hilbert::hilbert_transform_c2c(d_input, d_output);

	CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &input_resource));
	CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &output_resource));
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

	return true;
}

void
deinit_cuda_configuration()
{
	cleanup_session();
}
