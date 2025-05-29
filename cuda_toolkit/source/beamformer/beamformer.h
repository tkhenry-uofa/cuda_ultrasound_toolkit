#ifndef BEAMFORMER_H
#define BEAMFORMER_H
#include <span>

#include "../cuda_beamformer_parameters.h"
#include "../defs.h"


namespace beamform
{

    class Beamformer
    {
        public:
            Beamformer() = default;
            ~Beamformer() = default;

            bool beamform(std::span<const uint8_t> input_data, 
                          std::span<uint8_t> output_data, 
                          const CudaBeamformerParameters& params);
    };
}


#endif // BEAMFORMER_H