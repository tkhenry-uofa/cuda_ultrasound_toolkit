
#include "cuda_manager.h"
#include "cuda_toolkit.hpp"


static CudaManager& get_manager()
{
   static CudaManager manager;
   return manager;
}

bool 
cuda_toolkit::beamform(std::span<const uint8_t> input_data, 
                         std::span<uint8_t> output_data, 
                         const CudaBeamformerParameters& bp)
{
    auto& manager = get_manager();

    size_t raw_data_size = input_data.size();
    size_t output_data_size = output_data.size();

    void* d_input = nullptr;
    cuComplex* d_output = nullptr;

    CUDA_RETURN_IF_ERROR(cudaMalloc(&d_input, raw_data_size));
    CUDA_RETURN_IF_ERROR(cudaMalloc(&d_output, output_data_size));

    CUDA_RETURN_IF_ERROR(cudaMemcpy(d_input, input_data.data(), raw_data_size, cudaMemcpyHostToDevice));

    bool result = manager.beamform( d_input, d_output, bp );

    if (!result)
    {
        std::cerr << "Beamforming failed." << std::endl;
    }
    else
    {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(output_data.data(), d_output, output_data_size, cudaMemcpyDeviceToHost));
    }

   CUDA_RETURN_IF_ERROR(cudaFree(d_input));
   CUDA_RETURN_IF_ERROR(cudaFree(d_output));

   return result;
}
