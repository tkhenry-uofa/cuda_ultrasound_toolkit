#ifndef MAT_PARSER_H
#define MAT_PARSER_H


#include <vector>
#include <complex>
#include <string>

#include "defs.h"

extern "C" {
	#include <mat.h>
	#include <matrix.h>
}

namespace parser
{
	bool load_rf_data_array(std::string file_path, std::vector<float>** data_array, defs::RfDataDims* dims);

	bool save_float_array(float* ptr, size_t dims[3], std::string file_path, std::string variable_name);
}

#endif // !MAT_PARSER_H

