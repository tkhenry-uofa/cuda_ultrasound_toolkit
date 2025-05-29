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

            bool per_voxel_beamform(cuComplex* d_input,
                          cuComplex* d_volume,
                          const CudaBeamformerParameters& bp,
                          const float* hadamard);
    };
}


#endif // BEAMFORMER_H