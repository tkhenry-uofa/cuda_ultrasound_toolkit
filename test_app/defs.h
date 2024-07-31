#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <string.h>
#include <cufft.h>

#define THREADS_PER_BLOCK 512

#define SAMPLE_F = 50000000 // 50 MHz

#define MAX_ERROR_LENGTH 256
static char Error_buffer[MAX_ERROR_LENGTH];

#define FFT_RETURN_IF_ERROR(STATUS, MESSAGE)			\
{													\
	strcpy(Error_buffer, MESSAGE);					\
	strcat(Error_buffer, " Error code: %d.\n");		\
	if (STATUS != CUFFT_SUCCESS) {					\
		fprintf(stderr,Error_buffer,(int)STATUS);	\
		return false; }								\
}													\

typedef unsigned int uint;

namespace defs
{
	static const std::string rf_data_name = "rx_scans";

	struct ComplexF {
		float re = 0.0f;
		float im = 0.0f;
	};

	struct RfDataDims {
		size_t element_count;
		size_t sample_count;
		size_t tx_count;
	};

	enum Quadrent {
		TOP_LEFT = 0,
		TOP_RIGHT = 1,
		BOT_LEFT = 2,
		BOT_RIGHT = 3
	};
}


#endif // !DEFS_H
