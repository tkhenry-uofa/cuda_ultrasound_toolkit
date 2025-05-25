#ifndef MAT_PARSER_H
#define MAT_PARSER_H


#include <vector>
#include <complex>
#include <string>
#include <cuda_toolkit_testing.h>

#include "../defs.h"


extern "C" {
	#include <mat.h>
	#include <matrix.h>
}

namespace parser
{
	bool load_f2_tx_config(std::string file_path, PipelineParams* params);
	bool parse_bp_struct(std::string file_path, PipelineParams* params);
	bool load_int16_array(std::string file_path, std::vector<i16>** data_array, defs::RfDataDims* dims);

	bool load_float_array(std::string file_path, std::vector<float>** data_array, uint3* dims);
	bool load_complex_array(std::string file_path, std::vector<cuComplex>** data_array, uint3* dims);

	bool save_float_array(void* ptr, size_t dims[3], std::string file_path, std::string variable_name, bool complex);
}

#endif // !MAT_PARSER_H

