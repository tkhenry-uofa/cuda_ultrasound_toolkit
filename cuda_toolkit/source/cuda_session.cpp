#include "cuda_session.h"



bool
CudaSession::init(float2 rf_daw_dim, float3 dec_data_dim, InputDataType data_type)
{
    if (_init)
    {
        std::cout << "Session already initialized." << std::endl;
        return true;
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

    // Initialize cuBLAS and cuFFT
    cublasCreate(&_cublas_handle);
    cufftCreate(&_forward_plan);
    cufftCreate(&_inverse_plan);

    _init = true;
}

bool
CudaSession::register_ogl_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
    _unregister_ogl_buffers();

    cudaGraphicsResource_t raw_resource;
    CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&raw_resource, raw_data_ssbo, cudaGraphicsRegisterFlagsNone));
    _ogl_raw_buffers[raw_data_ssbo] = raw_resource;

    for (uint i = 0; i < rf_buffer_count; ++i)
    {
        cudaGraphicsResource_t rf_resource;
        CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&rf_resource, rf_data_ssbos[i], cudaGraphicsRegisterFlagsNone));
        _ogl_rf_buffers[rf_data_ssbos[i]] = rf_resource;
    }

    return true;
    
}


bool
CudaSession::_unregister_ogl_buffers()
{
    for (auto& pair : _ogl_raw_buffers)
    {
        cudaGraphicsResource_t resource = pair.second;
        if (resource)
        {
            CUDA_RETURN_IF_ERROR(cudaGraphicsUnregisterResource(resource));
        }
    }
    _ogl_raw_buffers.clear();

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
CudaSession::_setup_device_buffers()
{
    uint raw_type_size = 0;

    switch (_device_buffers.data_type)
    {
        case InputDataType::I16:
            raw_type_size = sizeof(i16);
            break;
        case InputDataType::F32:
            raw_type_size = sizeof(float);
            break;
        default:
            std::cerr << "Invalid data type." << std::endl;
            return false;
    }

    size_t raw_data_size = static_cast<size_t>(_rf_raw_dim.x * _rf_raw_dim.y) * raw_type_size;
    size_t decoded_data_size = static_cast<size_t>(_dec_data_dim.x * _dec_data_dim.y * _dec_data_dim.z) * sizeof(cuComplex);

    if (_device_buffers.raw_data_size != raw_data_size)
    {
        if (_device_buffers.d_raw)
        {
            CUDA_NULL_FREE(_device_buffers.d_raw);
        }
        _device_buffers.raw_data_size = raw_data_size;
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_raw, raw_data_size));
    }
    
    if (_device_buffers.decoded_data_size != decoded_data_size)
    {
        if (_device_buffers.d_decoded)
        {
            CUDA_NULL_FREE(_device_buffers.d_converted);
            CUDA_NULL_FREE(_device_buffers.d_decoded);
            CUDA_NULL_FREE(_device_buffers.d_hilbert);
        }
        _device_buffers.decoded_data_size = decoded_data_size;
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_converted, decoded_data_size));
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_decoded, decoded_data_size));
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_device_buffers.d_hilbert, decoded_data_size));
    }
}


bool
CudaSession::_cleanup_device_buffers()
{
    if (_device_buffers.d_raw)
    {
        CUDA_NULL_FREE(_device_buffers.d_raw);
    }
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

    _device_buffers.raw_data_size = 0;
    _device_buffers.decoded_data_size = 0;

    return true;
}