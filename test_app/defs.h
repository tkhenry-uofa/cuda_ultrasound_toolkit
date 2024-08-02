#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <string.h>
#include <cufft.h>

typedef unsigned int uint;
typedef int16_t i16;

namespace defs
{
	static const std::string rf_data_name = "rx_scans";

	struct ComplexF {
		float re = 0.0f;
		float im = 0.0f;
	};

    // Annotating Dim order
    typedef union
    {
        struct
        {
            uint sample_count;
            uint channel_count;
            uint tx_count;
        };
        struct uint3;

    } RfDataDims;
}


#endif // !DEFS_H

