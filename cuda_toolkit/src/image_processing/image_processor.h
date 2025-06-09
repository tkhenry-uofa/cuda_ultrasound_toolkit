#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <npp.h>
#include <cuda_runtime.h>
#include <span>

#include "../defs.h"


namespace image_processing 
{

    

class ImageProcessor
{

public:
    // Constructor
    ImageProcessor() {
        _stream_context = _create_stream_context();
    }

    bool svd_filter(std::span<const cuComplex> input, 
                    std::span<cuComplex> output, 
                    uint2 image_dims, 
                    std::span<const uint > mask);

private:

        // Creates the context for the default cuda stream
        NppStreamContext _create_stream_context();

        NppStreamContext _stream_context;

    
};


}




#endif // IMAGE_PROCESSOR_H