#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <span>

#include "../defs.h"


namespace image_processing 
{

class ImageProcessor
{
public:
    // Constructor
    ImageProcessor();


    bool svd_filter(std::span<const cuComplex> input, 
                    std::span<cuComplex> output, 
                    uint2 image_dims, 
                    std::span<const uint > mask);
};


}




#endif // IMAGE_PROCESSOR_H