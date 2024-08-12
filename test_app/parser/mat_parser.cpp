#include <iostream>

#include "mat_parser.h"

bool parser::parse_bp_struct(std::string file_path, defs::BeamformerParams* params)
{
    // Get RF Data

    MATFile* file = matOpen(file_path.c_str(), "r");

    mxArray* struct_array = matGetVariable(file, defs::beamformer_params_name.c_str());
    if (struct_array == NULL) {
        std::cerr << "Error reading tx configuration struct." << std::endl;
        return false;
    }

    mxArray* field_p = mxGetField(struct_array, 0, defs::channel_mapping_name.c_str());
    uint16_t* channel_map_mx = (uint16_t*)mxGetUint16s(field_p);
    for (uint i = 0; i < TOTAL_TOBE_CHANNELS; i++)
    {
        params->channel_mapping[i] = (uint)channel_map_mx[i];
    }

    field_p = mxGetField(struct_array, 0, defs::decoded_dims_name.c_str());

    mxArray* sub_field_p = mxGetField(field_p, 0, "x");
    double* double_mx = (double*)mxGetDoubles(sub_field_p);
    params->decoded_dims[0] = *double_mx;

    sub_field_p = mxGetField(field_p, 0, "y");
    double_mx = (double*)mxGetDoubles(sub_field_p);
    params->decoded_dims[1] = *double_mx;

    sub_field_p = mxGetField(field_p, 0, "z");
    double_mx = (double*)mxGetDoubles(sub_field_p);
    params->decoded_dims[2] = *double_mx;

    field_p = mxGetField(struct_array, 0, defs::raw_dims_name.c_str());
    sub_field_p = mxGetField(field_p, 0, "x");
    double_mx = (double*)mxGetDoubles(sub_field_p);
    params->raw_dims[0] = *double_mx;

    sub_field_p = mxGetField(field_p, 0, "y");
    double_mx = (double*)mxGetDoubles(sub_field_p);
    params->raw_dims[1] = *double_mx;

    field_p = mxGetField(struct_array, 0, defs::channel_offset_name.c_str());
    double_mx = (double*)mxGetDoubles(field_p);
    params->rx_cols = (*double_mx > 0);


    mxDestroyArray(struct_array);
    matClose(file);

}

bool
parser::load_int16_array(std::string file_path, std::vector<i16>** data_array, defs::RfDataDims* dims)
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
        size_t channel_count = mxGetNumberOfElements(mx_array);
        const mwSize* rf_size = mxGetDimensions(mx_array);
        dims->sample_count = rf_size[0];
        dims->channel_count = rf_size[1];
        dims->tx_count = rf_size[2];

        mxInt16* data_array_ptr = mxGetInt16s(mx_array);

        *data_array = new std::vector<i16>(data_array_ptr, &(data_array_ptr[channel_count]));

        success = true;
    }
    else
    {
        std::cerr << "Data is complex" << std::endl;
        return false;
    }

    mxDestroyArray(mx_array);
    matClose(file);

    return success;
}


bool
parser::load_float_array(std::string file_path, std::vector<float>** data_array, defs::RfDataDims* dims)
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
        size_t channel_count = mxGetNumberOfElements(mx_array);
        const mwSize* rf_size = mxGetDimensions(mx_array);
        dims->sample_count = rf_size[05];
        dims->channel_count = rf_size[1];
        dims->tx_count = rf_size[2];

        float* data_array_ptr = mxGetSingles(mx_array);

        *data_array = new std::vector<float>(data_array_ptr, &(data_array_ptr[channel_count]));

        success = true;
    }
    else
    {
        std::cerr << "Data is complex" << std::endl;
        return false;
    }

    mxDestroyArray(mx_array);
    matClose(file);

    return success;
}

bool
parser::save_float_array(void* ptr, size_t dims[3], std::string file_path, std::string variable_name, bool complex)
{

    mxArray* volume_array = NULL;
    if (complex)
    {
        volume_array = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxCOMPLEX);
        mxSetComplexSingles(volume_array, (mxComplexSingle*)ptr);
    }
    else
    {
        volume_array = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
        mxSetSingles(volume_array, (mxSingle*)ptr);
    }
    
    // Try to open for update, if it fails the file does not exist
    MATFile* file_p = matOpen(file_path.c_str(), "u");
    if (!file_p)
    {
        // Makes a new file or overwrites the old
        file_p = matOpen(file_path.c_str(), "w");
    }

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