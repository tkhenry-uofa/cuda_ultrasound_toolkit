#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <string.h>
#include <cufft.h>

typedef unsigned int uint;
typedef int16_t i16;

#define TOTAL_TOBE_CHANNELS 256



namespace defs
{
	static const std::string rf_data_name = "rx_scans";

    static const std::string beamformer_params_name = "bp";
    static const std::string channel_mapping_name = "channel_mapping";
    static const std::string decoded_dims_name = "dec_data_dim";
    static const std::string raw_dims_name = "rf_raw_dim";
    static const std::string channel_offset_name = "channel_offset";

    typedef struct {
        uint channel_mapping[256];
        uint decoded_dims[3];
        uint raw_dims[2];
        bool rx_cols;
    } BeamformerParams;

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

