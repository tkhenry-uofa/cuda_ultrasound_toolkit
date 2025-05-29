#include "cuda_manager.h"


CudaManager::CudaManager() : _init(false),
                            _rf_raw_dim({ 0, 0 }),
                            _dec_data_dim({ 0, 0, 0 }),
                            _decode_buffers(),
                            _beamformer_rf_buffer(nullptr)
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
CudaManager::init(uint2 rf_raw_dim, uint3 dec_data_dim)
{

    if (_init && !_dims_changed(rf_raw_dim, dec_data_dim))
    {
        std::cerr << "Session already initialized with the same dimensions." << std::endl;
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

    _rf_raw_dim = rf_raw_dim;
    _dec_data_dim = dec_data_dim;

    if(!_setup_decode_buffers())
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
CudaManager::deinit()
{
    if (!_init)
    {
        std::cerr << "Session not initialized." << std::endl;
        return false;
    }

    if(_beamformer_rf_buffer)
    {
        CUDA_NULL_FREE(_beamformer_rf_buffer);
        _beamformer_rf_buffer = nullptr;
    }

    _cleanup_decode_buffers();
    _init = false;

    return true;
}

bool 
CudaManager::set_channel_mapping(std::span<const int16_t> channel_mapping)
{
    if (!_data_converter->copy_channel_mapping(channel_mapping))
    {
        std::cerr << "Failed to load channel mapping." << std::endl;
        return false;
    }

    return true;
}

bool 
CudaManager::set_match_filter(std::span<const float> match_filter)
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
CudaManager::i16_convert_decode(i16* d_input, cuComplex* d_output)
{
    size_t data_count = _decoded_data_count();

    bool result = _data_converter->convert_i16(d_input, _decode_buffers.d_converted, _rf_raw_dim, _dec_data_dim);
    if (!result)
    {
        std::cerr << "Failed to convert data." << std::endl;
        return false;
    }

    result = _hadamard_decoder->decode(_decode_buffers.d_converted, _decode_buffers.d_decoded, _dec_data_dim);
    if (!result)
    {
        std::cerr << "Failed to decode Hadamard data." << std::endl;
        return false;
    }

    // Decode output is packed real floats, but OGL expects complex 
    // so do a strided copy to fit interleaved complex 
    cudaMemset(d_output, 0x00, data_count);
    CUDA_FLOAT_TO_COMPLEX_COPY(_decode_buffers.d_decoded, d_output, data_count);
    return true;
}

bool
CudaManager::hilbert_transform_strided(float* d_input, cuComplex* d_output)
{
    if (!_init)
    {
        std::cerr << "Session not initialized." << std::endl;
        return false;
    }

    bool result = _hilbert_handler->strided_hilbert_and_filter(d_input, d_output);
    if (!result)
    {
        std::cerr << "Failed to apply Hilbert transform and filter." << std::endl;
        return false;
    }

    return true;
}

bool
CudaManager::_setup_decode_buffers()
{
    size_t data_size = _decoded_data_count() * sizeof(float);

    if( data_size != _decode_buffers.size )
    {
        _decode_buffers.size = data_size;
        if (!_cleanup_decode_buffers())
        {
            std::cerr << "Failed to cleanup decode buffers." << std::endl;
            return false;
        }
    }
    
    CUDA_RETURN_IF_ERROR(cudaMalloc(&_decode_buffers.d_converted, data_size));
    CUDA_RETURN_IF_ERROR(cudaMalloc(&_decode_buffers.d_decoded, data_size));
 
    return true;
}

bool
CudaManager::_cleanup_decode_buffers()
{
    if (_decode_buffers.d_converted)
    {
        CUDA_NULL_FREE(_decode_buffers.d_converted);
    }
    if (_decode_buffers.d_decoded)
    {
        CUDA_NULL_FREE(_decode_buffers.d_decoded);
    }
    _decode_buffers.size = 0;
    return true;
}