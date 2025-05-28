#include "cuda_session.h"
#include "cuda_toolkit.h"

static CudaSession& get_session()
{
    static CudaSession session;
    return session;
}

bool
init_cuda_configuration(const uint* input_dims, const uint* decoded_dims)
{
    CudaSession& session = get_session();
    
    if (!session.init({input_dims[0], input_dims[1]}, {decoded_dims[0], decoded_dims[1], decoded_dims[2]}))
    {
        std::cerr << "Failed to initialize CUDA session." << std::endl;
        return false;
    }

    return true;
}

void
deinit_cuda_configuration()
{
    CudaSession& session = get_session();
    session.deinit();
}

bool
register_cuda_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
    CudaSession& session = get_session();
    
    std::span<const uint> rf_data_span(rf_data_ssbos, rf_buffer_count);
    if (!session.register_ogl_buffers(rf_data_span, raw_data_ssbo))
    {
        std::cerr << "Failed to register OpenGL buffers with CUDA." << std::endl;
        return false;
    }
    
    return true;
}

bool
cuda_set_channel_mapping(const i16 channel_mapping[MAX_CHANNEL_COUNT])
{
    CudaSession& session = get_session();
    
    std::span<const int16_t> mapping_span(channel_mapping, MAX_CHANNEL_COUNT);
    if (!session.set_channel_mapping(mapping_span))
    {
        std::cerr << "Failed to set channel mapping." << std::endl;
        return false;
    }
    
    return true;
}

bool
cuda_set_match_filter(const float* match_filter, uint length)
{
    CudaSession& session = get_session();
    
    std::span<const float> filter_span(match_filter, length);
    if (!session.set_match_filter(filter_span))
    {
        std::cerr << "Failed to set match filter." << std::endl;
        return false;
    }
    
    return true;
}

bool
cuda_decode(size_t input_offset, uint output_buffer_idx)
{
    CudaSession& session = get_session();
    if (!session.ogl_convert_decode(input_offset, output_buffer_idx))
    {
        std::cerr << "Failed to decode data." << std::endl;
        return false;
    }
    return true;
    
}

bool
cuda_hilbert(uint input_buffer_idx, uint output_buffer_idx)
{
    CudaSession& session = get_session();
    if (!session.ogl_hilbert(input_buffer_idx, output_buffer_idx))
    {
        std::cerr << "Failed to apply Hilbert transform." << std::endl;
        return false;
    }
    return true;
}

