
#include "beamformer/beamformer.h"
#include "rf_processing/rf_processor.h"
#include "cuda_toolkit.hpp"

static RfProcessor& get_rf_processor()
{
    static RfProcessor rf_processor;
    return rf_processor;
}

static Beamformer& get_beamformer()
{
   static Beamformer beamformer;
   return beamformer;
}

bool 
cuda_toolkit::beamform(std::span<const uint8_t> input_data, 
                         std::span<uint8_t> output_data, 
                         const CudaBeamformerParameters& bp)
{
    bool result = false;
    auto& beamformer = get_beamformer();
    auto& rf_processor = get_rf_processor();

    uint2 rf_raw_dim = { bp.rf_raw_dim[0], bp.rf_raw_dim[1] };
    uint3 dec_data_dim = { bp.dec_data_dim[0], bp.dec_data_dim[1], bp.dec_data_dim[2] };
    uint4 output_data_dim = { bp.output_points[0], bp.output_points[1], bp.output_points[2], bp.output_points[3] };


    size_t raw_data_size = input_data.size();
    size_t decoded_data_size = (size_t)(dec_data_dim.x) * dec_data_dim.y * dec_data_dim.z * sizeof(cuComplex);
    size_t output_data_size = output_data.size();

    void* d_input = nullptr;
    cuComplex* d_rf = nullptr;
    cuComplex* d_output = nullptr;

    CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, raw_data_size));
    CUDA_RETURN_IF_ERROR(cudaMalloc(&d_rf, decoded_data_size));
    CUDA_RETURN_IF_ERROR(cudaMalloc(&d_output, output_data_size));

    CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input_data.data(), raw_data_size, cudaMemcpyHostToDevice));

    std::span<const i16> channel_mapping(bp.channel_mapping, dec_data_dim.y);
    std::span<const float> rf_filter(bp.rf_filter, bp.filter_length);

    if(!rf_processor.init(rf_raw_dim, dec_data_dim))
    {
        std::cerr << "Failed to initialize RF processor." << std::endl;
        goto cleanup; // This is actually valid don't @ me
    }

    if(!rf_processor.set_channel_mapping(channel_mapping))
    {
        std::cerr << "Failed to set channel mapping." << std::endl;
        goto cleanup; // This is actually valid don't @ me
    }
    
    if(!rf_processor.set_match_filter(rf_filter))
    {
        std::cerr << "Failed to set match filter." << std::endl;
        goto cleanup; // This is actually valid don't @ me
    }

    if (!rf_processor.i16_convert_decode_strided(reinterpret_cast<i16*>(d_input), d_rf))
    {
        std::cerr << "Failed to decode RF data." << std::endl;
        goto cleanup; // This is actually valid don't @ me
    }
    
    if (!beamformer.beamform(d_rf, d_output, bp))
    {
        std::cerr << "Beamforming failed." << std::endl;
    }
    else
    {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(output_data.data(), d_output, output_data_size, cudaMemcpyDeviceToHost));
        result = true;
    }

cleanup:

   CUDA_RETURN_IF_ERROR(cudaFree(d_input));
   CUDA_RETURN_IF_ERROR(cudaFree(d_rf));
   CUDA_RETURN_IF_ERROR(cudaFree(d_output));

   return result;
}
