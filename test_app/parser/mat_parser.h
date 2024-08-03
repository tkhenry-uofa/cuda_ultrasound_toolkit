#ifndef MAT_PARSER_H
#define MAT_PARSER_H


#include <vector>
#include <complex>
#include <string>

#include "../defs.h"

extern "C" {
	#include <mat.h>
	#include <matrix.h>
}

namespace parser
{
	bool parse_bp_struct(std::string file_path, defs::BeamformerParams* params);
	bool load_int16_array(std::string file_path, std::vector<i16>** data_array, defs::RfDataDims* dims);

	bool load_float_array(std::string file_path, std::vector<float>** data_array, defs::RfDataDims* dims);

	bool save_float_array(void* ptr, size_t dims[3], std::string file_path, std::string variable_name, bool complex);
}

#endif // !MAT_PARSER_H

