#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <cufft.h>

#define RETURN_IF_ERROR(STATUS, MESSAGE)\
if (STATUS != CUFFT_SUCCESS) {            \
    fprintf(stderr, MESSAGE);  \
    return false;                      \
}   

typedef unsigned int uint;

namespace defs
{

	static const std::string rf_data_name = "rx_scans";

	struct RfDataDims {
		size_t element_count;
		size_t sample_count;
		size_t tx_count;
	};

}


#endif // !DEFS_H

