#include <iostream>

#include "mat_parser.h"

bool
parser::load_rf_data_array(std::string file_path, std::vector<float>** data_array, defs::RfDataDims* dims)
{
    
    bool success = false;
    mxArray* mx_array = nullptr;

    *data_array = nullptr;

    MATFile* file = matOpen(file_path.c_str(), "r");
 
    // Get RF Data
    mx_array = matGetVariable(file, defs::rf_data_name.c_str());
    if (mx_array == NULL) {
        std::cerr << "Error reading rf data array." << std::endl;
        return success;
    }

    if (!mxIsComplex(mx_array))
    {
        size_t element_count = mxGetNumberOfElements(mx_array);
        const mwSize* rf_size = mxGetDimensions(mx_array);
        dims->sample_count = rf_size[0];
        dims->element_count = rf_size[1];
        dims->tx_count = rf_size[2];

        float* data_array_ptr = mxGetSingles(mx_array);

        *data_array = new std::vector<float>(data_array_ptr, &(data_array_ptr[element_count]));

        success = true;
    }
    else
    {
        std::cerr << "Data is not complex" << std::endl;
        return false;
    }

    mxDestroyArray(mx_array);
    matClose(file);

    return success;
}

bool
parser::save_complex_data(std::complex<float>* ptr, size_t dims[3], std::string file_path, std::string variable_name)
{

    mxArray* volume_array = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxCOMPLEX);
    mxSetComplexSingles(volume_array, (mxComplexSingle*)ptr);

    MATFile* file_p = matOpen(file_path.c_str(), "w");
    if (!file_p)
    {
        std::cerr << "Failed to open file for volume: " << file_path << std::endl;
        mxDestroyArray(volume_array);
        return false;
    }

    int error = matPutVariable(file_p, variable_name.c_str(), volume_array);
    if (error)
    {
        std::cerr << "Failed to save array to file." << std::endl;
        return false;
    }

    matClose(file_p);

    return true;
}