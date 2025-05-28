#include "cuda_session.h"


CudaSession::CudaSession() : _init(false),
                            _rf_raw_dim({ 0, 0 }),
                            _dec_data_dim({ 0, 0, 0 }),
                            _device_buffers(),
                            _ogl_raw_buffer(),
                            _ogl_rf_buffers(),
	                        _beamformer_params()
{
    _data_converter = std::make_unique<data_conversion::DataConverter>();
    _hilbert_handler = std::make_unique<rf_fft::HilbertHandler>();
    _hadamard_decoder = std::make_unique<decoding::HadamardDecoder>();
    if (!_data_converter || !_hilbert_handler || !_hadamard_decoder)
    {
        std::cerr << "ERROR: Failed to setup CUDA resources." << std::endl;
    }
}

bool
CudaSession::init(uint2 rf_daw_dim, uint3 dec_data_dim)
{

    if (_init && !_dims_changed(rf_daw_dim, dec_data_dim))
    {
        std::cerr << "Session already initialized with the same dimensions and data type." << std::endl;
        return true;
    }

    int sample_count = static_cast<int>(dec_data_dim.x);
    int channel_count = static_cast<int>(dec_data_dim.y * dec_data_dim.z);

    if ( sample_count <= 0 || channel_count <= 0)
    {
        std::cerr << "Invalid dimensions for RF raw data: " << sample_count << " samples, " << channel_count << " channels." << std::endl;
        std::cerr << "Sample and channel count must both fit under int32 max." << std::endl;
        return false;
    }

    _rf_raw_dim = rf_daw_dim;
    _dec_data_dim = dec_data_dim;

    if(!_setup_device_buffers())
    {
        std::cerr << "Failed to setup buffers." << std::endl;
        return false;
    }

    // Initialize CUDA resources
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    if (!_hadamard_decoder->generate_hadamard(dec_data_dim.z, ReadiOrdering::HADAMARD))
    {
        std::cerr << "Failed to generate Hadamard matrix." << std::endl;
        return false;
    }

    if (!_hilbert_handler->plan_ffts({ sample_count, channel_count }))
    {
        std::cerr << "Failed to plan FFTs." << std::endl;
        return false;
    }

    _init = true;
    return true;
}

bool
CudaSession::deinit()
{
    if (!_init)
    {
        std::cerr << "Session not initialized." << std::endl;
        return false;
    }

    unregister_ogl_buffers();
    _cleanup_device_buffers();
    _init = false;

    return true;
}

bool 
CudaSession::set_channel_mapping(std::span<const int16_t> channel_mapping)
{
    if (!_data_converter->copy_channel_mapping(channel_mapping))
    {
        std::cerr << "Failed to load channel mapping." << std::endl;
        return false;
    }

    return true;
}

bool 
CudaSession::set_match_filter(std::span<const float> match_filter)
{
    if (!_init)
    {
        std::cerr << "Session not initialized." << std::endl;
        return false;
    }

    if (match_filter.empty())
    {
        std::cerr << "Match filter cannot be empty." << std::endl;
        return false;
    }

    if (!_hilbert_handler->load_filter(match_filter))
    {
        std::cerr << "Failed to load match filter." << std::endl;
        return false;
    }

    return true;
}

bool
CudaSession::register_ogl_buffers(std::span<const uint> rf_data_ssbos, uint raw_data_ssbo)
{
    unregister_ogl_buffers();

    cudaGraphicsResource_t raw_resource;
    CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&raw_resource, raw_data_ssbo, cudaGraphicsRegisterFlagsNone));

    _ogl_raw_buffer = { raw_data_ssbo, raw_resource };

    for (const auto& ssbo : rf_data_ssbos)
    {
        cudaGraphicsResource_t rf_resource;
        CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&rf_resource, ssbo, cudaGraphicsRegisterFlagsNone));
        _ogl_rf_buffers.push_back({ ssbo, rf_resource });
    }

    return true;
    
}

bool
CudaSession::unregister_ogl_buffers()
{

    if (_ogl_raw_buffer.second)
    {
        CUDA_RETURN_IF_ERROR(cudaGraphicsUnregisterResource(_ogl_raw_buffer.second));
    }

    for (auto& pair : _ogl_rf_buffers)
    {
        cudaGraphicsResource_t resource = pair.second;
        if (resource)
        {
            CUDA_RETURN_IF_ERROR(cudaGraphicsUnregisterResource(resource));
        }
    }
    _ogl_rf_buffers.clear();
    
    return true;
}

bool
CudaSession::ogl_convert_decode(size_t raw_buffer_offset, uint output_buffer_idx)
{
    if (!_init)
    {
        std::cerr << "Session not initialized." << std::endl;
        return false;
    }

    auto output_resource = _ogl_rf_buffers[output_buffer_idx].second;

    // Right now we only support one input buffer
    auto input_resource = _ogl_raw_buffer.second;

    i16* d_input = nullptr;
    cuComplex* d_output = nullptr;

    // From this point we MUST unmap the buffers before exiting, hence the goto's 
    _map_ogl_buffer((void**)&d_input, input_resource);
    _map_ogl_buffer((void**)&d_output, output_resource);

    d_input = d_input + raw_buffer_offset / sizeof(i16);
    size_t data_count = _decoded_data_count();

    bool result = _data_converter->convert_i16(d_input, _device_buffers.d_converted, _rf_raw_dim, _dec_data_dim);
    if (!result)
    {
        std::cerr << "Failed to convert data." << std::endl;
        goto unmap; // I promise this is a valid use
    }
    result = _hadamard_decoder->decode(_device_buffers.d_converted, _device_buffers.d_decoded, _dec_data_dim);
    if (!result)
    {
        std::cerr << "Failed to decode Hadamard data." << std::endl;
        goto unmap;
    }
    // Decode output is packed real floats, but OGL expects complex 
    // so do a strided copy to fit interleaved complex 
    cudaMemset(d_output, 0x00, data_count);
    CUDA_FLOAT_TO_COMPLEX_COPY(_device_buffers.d_decoded, d_output, data_count);

unmap:
    _unmap_ogl_buffer(input_resource);
    _unmap_ogl_buffer(output_resource);

    return result;
}

bool
CudaSession::ogl_hilbert(uint intput_buffer_idx, uint output_buffer_idx)
{
    if (!_init)
    {
        std::cerr << "Session not initialized." << std::endl;
        return false;
    }

    auto input_resource = _ogl_rf_buffers[intput_buffer_idx].second;
    auto output_resource = _ogl_rf_buffers[output_buffer_idx].second;

    cuComplex* d_input = nullptr;
    cuComplex* d_output = nullptr; 

    // From this point we MUST unmap the buffers before exiting 
    _map_ogl_buffer((void**)&d_input, input_resource);
    _map_ogl_buffer((void**)&d_output, output_resource);

    bool result = _hilbert_handler->strided_hilbert_and_filter(d_input, d_output);
    if (!result)
    {
        std::cerr << "Failed to apply Hilbert transform and filter." << std::endl;
    }

    _unmap_ogl_buffer(input_resource);
    _unmap_ogl_buffer(output_resource);

    return result;
}

inline bool
CudaSession::_map_ogl_buffer(void** d_ptr, cudaGraphicsResource_t ogl_resource)
{
    if (!ogl_resource)
    {
        std::cerr << "OpenGL resource is null." << std::endl;
        return false;
    }

    size_t num_bytes;
    CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &ogl_resource));
    CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer(d_ptr, &num_bytes, ogl_resource));
    
    if (*d_ptr == nullptr)
    {
        std::cerr << "Failed to map OpenGL buffer." << std::endl;
        return false;
    }

    return true;
}

bool
CudaSession::_setup_device_buffers()
{
    uint raw_type_size = 0;

    size_t float_data_size = (size_t)(_dec_data_dim.x) * _dec_data_dim.y * _dec_data_dim.z * sizeof(float);
    size_t cplx_data_size = float_data_size * 2;
    
    if (_device_buffers.decoded_data_size != float_data_size)
    {
        if (_device_buffers.d_decoded)
        {
            CUDA_NULL_FREE(_device_buffers.d_converted);
            CUDA_NULL_FREE(_device_buffers.d_decoded);
            CUDA_NULL_FREE(_device_buffers.d_hilbert);
        }
        _device_buffers.decoded_data_size = float_data_size;
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_converted, float_data_size));
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_decoded, float_data_size));
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_hilbert, cplx_data_size));
    }

    return true;
}

bool
CudaSession::_cleanup_device_buffers()
{
    if (_device_buffers.d_converted)
    {
        CUDA_NULL_FREE(_device_buffers.d_converted);
    }
    if (_device_buffers.d_decoded)
    {
        CUDA_NULL_FREE(_device_buffers.d_decoded);
    }
    if (_device_buffers.d_hilbert)
    {
        CUDA_NULL_FREE(_device_buffers.d_hilbert);
    }
    if( _device_buffers.d_volume)
    {
        CUDA_NULL_FREE(_device_buffers.d_volume);
    }

    _device_buffers.decoded_data_size = 0;

    return true;
}