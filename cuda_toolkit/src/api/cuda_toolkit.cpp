
#include "../beamformer/beamformer.h"
#include "../rf_processing/rf_processor.h"
#include "../public/cuda_toolkit.hpp"

struct BeamformerBuffers
{
    void* d_input = nullptr;
    cuComplex* d_decoded = nullptr;
    cuComplex* d_rf = nullptr;
    cuComplex* d_output = nullptr;
    size_t raw_data_size = 0;
    size_t decoded_data_size = 0;
    size_t output_data_size = 0;
};

static BeamformerBuffers& get_beamformer_buffers()
{
    static BeamformerBuffers buffers;
    return buffers;
}

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


static bool
size_changed(size_t rf, size_t dec, size_t out)
{
	auto& buffers = get_beamformer_buffers();
	return buffers.raw_data_size != rf ||
		buffers.decoded_data_size != dec ||
		buffers.output_data_size != out;
}

static void
cleanup_beamformer_buffers()
{
	auto& buffers = get_beamformer_buffers();
	if (buffers.d_input) {
		cudaFree(buffers.d_input);
		buffers.d_input = nullptr;
	}
	if (buffers.d_decoded) {
		cudaFree(buffers.d_decoded);
		buffers.d_decoded = nullptr;
	}
	if (buffers.d_rf) {
		cudaFree(buffers.d_rf);
		buffers.d_rf = nullptr;
	}
	if (buffers.d_output) {
		cudaFree(buffers.d_output);
		buffers.d_output = nullptr;
	}
	buffers.raw_data_size = 0;
	buffers.decoded_data_size = 0;
	buffers.output_data_size = 0;
}

static bool
update_beamformer_buffers(size_t rf, size_t dec, size_t out)
{
	auto& buffers = get_beamformer_buffers();
	if (size_changed(rf, dec, out)) {
        cleanup_beamformer_buffers();

		CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers.d_input, rf));
		CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers.d_decoded, dec));
		CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers.d_rf, dec));
		CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers.d_output, out));
        buffers.raw_data_size = rf;
        buffers.decoded_data_size = dec;
        buffers.output_data_size = out;
	}
    return true;
}

static void
inspect_buffers()
{
	auto& buffers = get_beamformer_buffers();

	void* input = std::malloc(buffers.raw_data_size);
	cuComplex* decoded = (cuComplex*)std::malloc(buffers.decoded_data_size);
	cuComplex* rf = (cuComplex*)std::malloc(buffers.decoded_data_size);
	cuComplex* output = (cuComplex*)std::malloc(buffers.output_data_size);

	cudaMemcpy(input, buffers.d_input, buffers.raw_data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(decoded, buffers.d_decoded, buffers.decoded_data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(rf, buffers.d_rf, buffers.decoded_data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, buffers.d_output, buffers.output_data_size, cudaMemcpyDeviceToHost);

	std::cout << "Input buffer size: " << buffers.raw_data_size << " bytes" << std::endl;
	std::cout << "Decoded buffer size: " << buffers.decoded_data_size << " bytes" << std::endl;
	std::cout << "RF buffer size: " << buffers.decoded_data_size << " bytes" << std::endl;

	free(input);
	free(decoded);
	free(rf);
	free(output);
}

bool 
cuda_toolkit::beamform(std::span<const uint8_t> input_data, 
                         std::span<uint8_t> output_data, 
                         const CudaBeamformerParameters& bp)
{
    bool result = false;
    auto& beamformer = get_beamformer();
    auto& rf_processor = get_rf_processor();
	auto& buffers = get_beamformer_buffers();

    uint2 rf_raw_dim =      { bp.rf_raw_dim[0], bp.rf_raw_dim[1] };
    uint3 dec_data_dim =    { bp.dec_data_dim[0], bp.dec_data_dim[1], bp.dec_data_dim[2] };
    uint4 output_data_dim = { bp.output_points[0], bp.output_points[1], bp.output_points[2], bp.output_points[3] };

    size_t raw_data_size = input_data.size();
    size_t decoded_data_size = (size_t)(dec_data_dim.x) * dec_data_dim.y * dec_data_dim.z * sizeof(cuComplex);
    size_t output_data_size = output_data.size();

	if (!update_beamformer_buffers(raw_data_size, decoded_data_size, output_data_size))
	{
		std::cerr << "Failed to update beamformer buffers." << std::endl;
		return false; // Failed to allocate buffers
	}

    CUDA_RETURN_IF_ERROR(cudaMemcpy(buffers.d_input, input_data.data(), raw_data_size, cudaMemcpyHostToDevice));

    std::span<const i16> channel_mapping(bp.channel_mapping, dec_data_dim.y);
    std::span<const float> rf_filter(bp.rf_filter, bp.filter_length);
    if(!rf_processor.init(rf_raw_dim, dec_data_dim))
    {
        std::cerr << "Failed to initialize RF processor." << std::endl;
        return false;
    }

    if(!rf_processor.set_channel_mapping(channel_mapping))
    {
        std::cerr << "Failed to set channel mapping." << std::endl;
        return false;
    }
    
    if(!rf_processor.set_match_filter(rf_filter))
    {
        std::cerr << "Failed to set match filter." << std::endl;
        return false;
    }

    if (!rf_processor.convert_decode_strided(buffers.d_input, buffers.d_decoded, bp.data_type))
    {
        std::cerr << "Failed to decode RF data." << std::endl;
        return false;
    }

	if (!rf_processor.hilbert_transform_strided((float*)buffers.d_decoded, buffers.d_rf))
	{
		std::cerr << "Failed to apply Hilbert transform." << std::endl;
        return false;
	}
    
    if (beamformer.beamform(buffers.d_rf, buffers.d_output, bp))
    {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(output_data.data(), buffers.d_output, output_data_size, cudaMemcpyDeviceToHost));
		inspect_buffers(); // Optional: Inspect buffers for debugging
        return true;
    }
    else
    {
        std::cerr << "Beamforming failed." << std::endl;
        return false;
    }
}
