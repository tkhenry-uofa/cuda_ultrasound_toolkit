#include "defs.h"
#include "session.h"
#include "filters/hilbert_transform.cuh"
#include "filters/sparse_filters.h"
#include "decoding/hadamard.cuh"
#include "decoding/int16_to_float.cuh"
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
	if (Session.raw_data_ssbo.cuda_resource != nullptr)
	{
		CUDA_RETURN_IF_ERR(cudaGraphicsUnregisterResource(Session.raw_data_ssbo.cuda_resource));
		Session.raw_data_ssbo.cuda_resource = nullptr;
	}

	uint old_buffer_count = Session.rf_buffer_count;
	if (old_buffer_count != 0)
	{
		for (uint i = 0; i < old_buffer_count; i++)
		{
			CUDA_RETURN_IF_ERR(cudaGraphicsUnregisterResource(Session.rf_data_ssbos[i].cuda_resource));
		}

		free(Session.rf_data_ssbos);
	}
	Session.rf_buffer_count = 0;

	return true;
}

bool
_cleanup_session()
{
	CUDA_NULL_FREE(Session.d_complex);
	CUDA_NULL_FREE(Session.d_hadamard);
	CUDA_NULL_FREE(Session.d_converted);
	CUDA_NULL_FREE(Session.d_decoded);
	CUDA_NULL_FREE(Session.d_input);
	cublasDestroy(Session.cublas_handle);
	Session.cublas_handle = nullptr;
	cusparseDestroy(Session.cusparse_handle);
	Session.cusparse_handle = nullptr;
	cufftDestroy(Session.forward_plan);
	cufftDestroy(Session.inverse_plan);

	cusparseDestroyDnMat(Session.d_decoded_dense);
	cusparseDestroyDnMat(Session.d_spectrum_dense);
	if (Session.channel_mapping)
	{
		free(Session.channel_mapping);
		Session.channel_mapping = nullptr;
	}
	Session.init = false;

	delete Session.hilbert_sparse;
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

		CUBLAS_RETURN_IF_ERR(cublasCreate(&(Session.cublas_handle)));
		CUSPARSE_RETURN_IF_ERR(cusparseCreate(&(Session.cusparse_handle)));
		CUFFT_RETURN_IF_ERR(cufftCreate(&(Session.forward_plan)));
		CUFFT_RETURN_IF_ERR(cufftCreate(&(Session.inverse_plan)));

		size_t input_size = input_dims[0] * input_dims[1];
		size_t decoded_size = decoded_dims[0] * decoded_dims[1] * decoded_dims[2];

		//std::cout << "Allocing memory" << std::endl;
		CUDA_RETURN_IF_ERR(cudaMalloc((void**)&(Session.d_input), input_size * sizeof(i16)));
		CUDA_RETURN_IF_ERR(cudaMalloc((void**)&(Session.d_converted), decoded_size * sizeof(float)));
		CUDA_RETURN_IF_ERR(cudaMalloc((void**)&(Session.d_decoded), decoded_size * sizeof(float)));
		CUDA_RETURN_IF_ERR(cudaMalloc((void**)&(Session.d_complex), decoded_size * sizeof(cuComplex)));

		bool success = hadamard::generate_hadamard(decoded_dims[2], &(Session.d_hadamard));
		ASSERT(success);


		// Create matrix for hilbert transform
		SparseMatrix* hilbert_mat = sparse_filters::create_hilbert_filter(Session.decoded_dims.x);

		if (!hilbert_mat)
		{
			return false;
		}
		Session.hilbert_sparse = hilbert_mat;

		// Get cusparse descriptors for the decoded and spectrum arrays

		CUSPARSE_RETURN_IF_ERR(cusparseCreateDnMat(&(Session.d_decoded_dense), 
			Session.decoded_dims.x, 
			Session.decoded_dims.y * Session.decoded_dims.z, 
			0, 
			(cuComplex*)(Session.d_decoded), 
			CUDA_C_32F, 
			CUSPARSE_ORDER_ROW));
		
		CUSPARSE_RETURN_IF_ERR(cusparseCreateDnMat(&(Session.d_spectrum_dense),
			Session.decoded_dims.x,
			Session.decoded_dims.y * Session.decoded_dims.z,
			0,
			Session.d_spectrum,
			CUDA_C_32F,
			CUSPARSE_ORDER_ROW));

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
	if (!Session.init)
	{
		return _init_session(input_dims, decoded_dims, channel_mapping, rx_cols);
	}
	else
	{
		bool changed = input_dims[0] != Session.input_dims.x || input_dims[1] != Session.input_dims.y ||
					   decoded_dims[0] != Session.decoded_dims.x || decoded_dims[1] != Session.decoded_dims.y || 
					   decoded_dims[2] != Session.decoded_dims.z;

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
	
	Session.raw_data_ssbo = { nullptr, 0 };
	Session.raw_data_ssbo.gl_buffer_id = raw_data_ssbo;
	CUDA_RETURN_IF_ERR(cudaGraphicsGLRegisterBuffer(&(Session.raw_data_ssbo.cuda_resource), Session.raw_data_ssbo.gl_buffer_id, cudaGraphicsRegisterFlagsNone));

	Session.rf_data_ssbos = (BufferMapping*)malloc(rf_buffer_count * sizeof(BufferMapping));
	for (uint i = 0; i < rf_buffer_count; i++)
	{
		std::cout << "Registering buffer : " << i << ", " << rf_data_ssbos[i] << std::endl;
		Session.rf_data_ssbos[i] = { NULL, rf_data_ssbos[i] };
		CUDA_RETURN_IF_ERR(cudaGraphicsGLRegisterBuffer(&(Session.rf_data_ssbos[i].cuda_resource), Session.rf_data_ssbos[i].gl_buffer_id, cudaGraphicsRegisterFlagsNone));
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

	std::cout << "Decoding, input off: " << input_offset << " bufferId: " << Session.rf_data_ssbos[output_buffer_idx].gl_buffer_id << std::endl;

	cudaGraphicsResource_t input_resource = Session.raw_data_ssbo.cuda_resource;
	cudaGraphicsResource_t output_resource = Session.rf_data_ssbos[output_buffer_idx].cuda_resource;
	if (!input_resource || !output_resource)
	{
		fprintf(stderr, "Open GL buffers not registered with cuda.");
		return false;
	}

	CUDA_RETURN_IF_ERR(cudaGraphicsMapResources(1, &input_resource));
	CUDA_RETURN_IF_ERR(cudaGraphicsMapResources(1, &output_resource));
	size_t total_count = Session.decoded_dims.x * Session.decoded_dims.y * Session.decoded_dims.z;

	size_t num_bytes;
	i16* d_input = nullptr;
	cufftComplex* d_output = nullptr;
	CUDA_RETURN_IF_ERR(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &num_bytes, input_resource));
	CUDA_RETURN_IF_ERR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));
	CUDA_RETURN_IF_ERR(cudaDeviceSynchronize());

	d_input = d_input + input_offset / sizeof(i16);

	i16_to_f::convert_data(d_input, Session.d_converted, Session.rx_cols);
	hadamard::hadamard_decode(Session.d_converted, Session.d_decoded);

	// Insert 0s between each value for their imaginary components
	CUDA_RETURN_IF_ERR(cudaMemset(d_output, 0x00, total_count * sizeof(cuComplex)));
	CUDA_RETURN_IF_ERR(cudaMemcpy2D(d_output, 2 * sizeof(float), Session.d_decoded, sizeof(float), sizeof(float), total_count, cudaMemcpyDefault));

	CUDA_RETURN_IF_ERR(cudaGraphicsUnmapResources(1, &input_resource));
	CUDA_RETURN_IF_ERR(cudaGraphicsUnmapResources(1, &output_resource));
	CUDA_RETURN_IF_ERR(cudaDeviceSynchronize());

	std::cout << "First input value: " << sample_value_i16(d_input) << std::endl;
	std::cout << "First decoded value: " << sample_value((float*)d_output) << std::endl;

	return true;
}

bool
cuda_hilbert(uint input_buffer_idx, uint output_buffer_idx)
{
	std::cout << "Hilbert, inputId: " << Session.rf_data_ssbos[input_buffer_idx].gl_buffer_id << " outputId: " << Session.rf_data_ssbos[output_buffer_idx].gl_buffer_id << std::endl;
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

	CUDA_RETURN_IF_ERR(cudaGraphicsMapResources(1, &input_resource));
	CUDA_RETURN_IF_ERR(cudaGraphicsMapResources(1, &output_resource));

	size_t num_bytes;
	cuComplex* d_input = nullptr;
	cuComplex* d_output = nullptr;
	CUDA_RETURN_IF_ERR(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &num_bytes, input_resource));
	CUDA_RETURN_IF_ERR(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, output_resource));

	hilbert::hilbert_transformC2C(d_input, d_output);

	std::cout << "First input value: " << sample_value((float*)d_input) << std::endl;
	std::cout << "First output value: " << sample_value((float*)d_output) << std::endl;

	CUDA_RETURN_IF_ERR(cudaGraphicsUnmapResources(1, &input_resource));
	CUDA_RETURN_IF_ERR(cudaGraphicsUnmapResources(1, &output_resource));
	CUDA_RETURN_IF_ERR(cudaDeviceSynchronize());

	return true;
}
