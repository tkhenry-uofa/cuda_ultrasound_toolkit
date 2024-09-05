#include <iostream>
#include <stdexcept>

#include "mat_parser.h"

bool parser::parse_bp_struct(std::string file_path, BeamformerParams* params)
{
    MATFile* file = matOpen(file_path.c_str(), "r");

    mxArray* struct_array = matGetVariable(file, defs::beamformer_params_name.c_str());
    if (struct_array == NULL) {
        std::cerr << "Error reading tx configuration struct." << std::endl;
        return false;
    }

    mxArray* field_p = mxGetField(struct_array, 0, defs::channel_mapping_name.c_str());

    if (field_p)
    {
        uint16_t* channel_map_mx = (uint16_t*)mxGetUint16s(field_p);
        if (!channel_map_mx)
        {
            double* channel_map_d = (double*)mxGetDoubles(field_p);
            if (!channel_map_d)
            {
                std::cout << "Failed to read " << defs::channel_mapping_name << " invalid data type." << std::endl;
            }
            else
            {
                for (uint i = 0; i < TOTAL_TOBE_CHANNELS; i++)
                {
                    params->channel_mapping[i] = (uint)channel_map_d[i];
                }
            }
        }
        else
        {
            for (uint i = 0; i < TOTAL_TOBE_CHANNELS; i++)
            {
                params->channel_mapping[i] = (uint)channel_map_mx[i];
            }
        }


    }
    else
    {
        std::cout << "Failed to read " << defs::channel_mapping_name << std::endl;
    }

    field_p = mxGetField(struct_array, 0, defs::decoded_dims_name.c_str());
    mxArray* sub_field_p;
    double* double_mx;
    if (field_p)
    {
        sub_field_p = mxGetField(field_p, 0, "x");
        double_mx = (double*)mxGetDoubles(sub_field_p);
        params->decoded_dims[0] = (uint)*double_mx;

        sub_field_p = mxGetField(field_p, 0, "y");
        double_mx = (double*)mxGetDoubles(sub_field_p);
        params->decoded_dims[1] = (uint)*double_mx;

        sub_field_p = mxGetField(field_p, 0, "z");
        double_mx = (double*)mxGetDoubles(sub_field_p);
        params->decoded_dims[2] = (uint)*double_mx;
    }
    else
    {
        std::cout << "Failed to read " << defs::decoded_dims_name << std::endl;
    }



    field_p = mxGetField(struct_array, 0, defs::raw_dims_name.c_str());
    if (field_p)
    {
        sub_field_p = mxGetField(field_p, 0, "x");
        double_mx = (double*)mxGetDoubles(sub_field_p);
        params->raw_dims[0] = (uint)*double_mx;

        sub_field_p = mxGetField(field_p, 0, "y");
        double_mx = (double*)mxGetDoubles(sub_field_p);
        params->raw_dims[1] = (uint)*double_mx;
    }
    else
    {
        std::cout << "Failed to read " << defs::raw_dims_name << std::endl;
    }

    field_p = mxGetField(struct_array, 0, defs::channel_offset_name.c_str());
    if (field_p)
    {
        double_mx = (double*)mxGetDoubles(field_p);
        params->rx_cols = (*double_mx > 0);
    }
    else
    {
        std::cout << "Failed to read " << defs::channel_offset_name << std::endl;
    }

    field_p = mxGetField(struct_array, 0, defs::center_freq_name.c_str());
    if (field_p)
    {
        double_mx = (double*)mxGetDoubles(field_p);
        params->array_params.center_freq = (float)*double_mx;
    }
    else
    {
        std::cout << "Failed to read " << defs::center_freq_name << std::endl;
    }

    field_p = mxGetField(struct_array, 0, defs::sample_freq_name.c_str());
    if (field_p)
    {
        double_mx = (double*)mxGetDoubles(field_p);
        params->array_params.sample_freq = (float)*double_mx;
    }
    else
    {
        std::cout << "Failed to read " << defs::sample_freq_name << std::endl;
    }

    field_p = mxGetField(struct_array, 0, defs::pitch_name.c_str());
    if (field_p)
    {
        double_mx = (double*)mxGetDoubles(field_p);
        params->array_params.pitch = (float)*double_mx;
    }
    else
    {
        std::cout << "Failed to read " << defs::pitch_name << std::endl;
    }

    field_p = mxGetField(struct_array, 0, defs::focus_name.c_str());
    if (field_p)
    {
        sub_field_p = mxGetField(field_p, 0, "x");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->focus[0] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read focus x" << std::endl;
        }

        sub_field_p = mxGetField(field_p, 0, "y");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->focus[1] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read focus y" << std::endl;
        }

        sub_field_p = mxGetField(field_p, 0, "z");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->focus[2] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read focus z" << std::endl;
        }

    }
    else
    {
        // Only the depth is specified
        field_p = mxGetField(struct_array, 0, defs::focal_depth.c_str());
        if (field_p)
        {
            double_mx = (double*)mxGetDoubles(field_p);
            params->focus[2] = (float)*double_mx;

            params->focus[0] = 0.0f;
            params->focus[1] = 0.0f;
        }
        else
        {
            std::cout << "Failed to read focus" << std::endl;
        }

    }

    field_p = mxGetField(struct_array, 0, defs::xdc_min_name.c_str());
    if (field_p)
    {
        sub_field_p = mxGetField(field_p, 0, "x");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->array_params.xdc_mins[0] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read xdc min x" << std::endl;
        }

        sub_field_p = mxGetField(field_p, 0, "y");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->array_params.xdc_mins[1] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read xdc min y" << std::endl;
        }

    }

    field_p = mxGetField(struct_array, 0, defs::xdc_max_name.c_str());
    if (field_p)
    {
        sub_field_p = mxGetField(field_p, 0, "x");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->array_params.xdc_maxes[0] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read xdc max x" << std::endl;
        }

        sub_field_p = mxGetField(field_p, 0, "y");
        if (sub_field_p)
        {
            double_mx = (double*)mxGetDoubles(sub_field_p);
            params->array_params.xdc_maxes[1] = (float)*double_mx;
        }
        else
        {
            std::cout << "Failed to read xdc max y" << std::endl;
        }

    }

    mxDestroyArray(struct_array);
    matClose(file);

    return true;
}

bool
parser::load_f2_tx_config(std::string file_path, BeamformerParams* params)
{
    bool success = false;

    MATFile* file = matOpen(file_path.c_str(), "r");

    mxArray* struct_array = matGetVariable(file, defs::sims::tx_config_name.c_str());
    if (struct_array == NULL) {
        std::cerr << "Error reading tx configuration struct." << std::endl;
        ASSERT(false);
        return false;
    }


    // TODO: Catch log and throw null returns
    mxArray* field_p = mxGetField(struct_array, 0, defs::sims::f0_name.c_str());
    params->array_params.center_freq = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::sims::fs_name.c_str());
    params->array_params.sample_freq = (float)*mxGetDoubles(field_p);


    field_p = mxGetField(struct_array, 0, defs::sims::col_count_name.c_str());
    params->array_params.col_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::sims::row_count_name.c_str());
    params->array_params.row_count = (int)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::sims::pitch_name.c_str());
    params->array_params.pitch = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::sims::x_min_name.c_str());
    params->array_params.xdc_mins[0] = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::sims::x_max_name.c_str());
    params->array_params.xdc_maxes[0] = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::sims::y_min_name.c_str());
    params->array_params.xdc_mins[1] = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::sims::y_max_name.c_str());
    params->array_params.xdc_maxes[1] = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::sims::tx_count_name.c_str());
    params->decoded_dims[2] = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::sims::pulse_delay_name.c_str());
    params->pulse_delay = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::sims::focus_name.c_str());
    double* src_locs = (double*)mxGetDoubles(field_p);
    params->focus[0] = src_locs[0];
    params->focus[1] = src_locs[1];
    params->focus[2] = src_locs[2];


    success = true;
    return success;
}


bool
parser::load_int16_array(std::string file_path, std::vector<i16>** data_array, defs::RfDataDims* dims)
{
    bool success = false;
    mxArray* mx_array = nullptr;

    *data_array = nullptr;

    MATFile* file = matOpen(file_path.c_str(), "r");

    if (!file)
    {
        std::cerr << "Cannot open file: " << file_path << std::endl;
        return success;
    }

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
        dims->sample_count = (uint)rf_size[0];
        dims->channel_count = (uint)rf_size[1];
        dims->tx_count = (uint)rf_size[2];

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
parser::load_float_array(std::string file_path, std::vector<float>** data_array, uint3* dims)
{

    mxArray* mx_array = nullptr;

    *data_array = nullptr;

    MATFile* file = matOpen(file_path.c_str(), "r");

    if (!file)
    {
        std::cerr << "Input file not found: " + file_path << std::endl;
        ASSERT(false);
        return false;
    }

    // Get RF Data
    mx_array = matGetVariable(file, defs::sims::rf_data_name.c_str());
    if (mx_array == NULL) {
        std::cerr << "Error reading rf data array." << std::endl;
        ASSERT(false);
        return false;
    }

    if (mxIsComplex(mx_array))
    {
        std::cerr << "Data is complex" << std::endl;
        ASSERT(false);
        return false;
    }

    size_t element_count = mxGetNumberOfElements(mx_array);
    const mwSize* rf_size = mxGetDimensions(mx_array);
    dims->x = (uint)rf_size[0];
    dims->y = (uint)rf_size[1];
    dims->x = (uint)rf_size[2];

    float* data_array_ptr = mxGetSingles(mx_array);

    *data_array = new std::vector<float>(data_array_ptr, &(data_array_ptr[element_count]));

    mxDestroyArray(mx_array);
    matClose(file);

    return true;
}

bool
parser::load_complex_array(std::string file_path, std::vector<cuComplex>** data_array, uint3* dims)
{

    mxArray* mx_array = nullptr;

    *data_array = nullptr;

    MATFile* file = matOpen(file_path.c_str(), "r");

    if (!file)
    {
        std::cerr << "Input file not found: " + file_path << std::endl;
        ASSERT(false);
        return false;
    }

    // Get RF Data
    mx_array = matGetVariable(file, defs::sims::rf_data_name.c_str());
    if (mx_array == NULL) {
        std::cerr << "Error reading rf data array." << std::endl;
        ASSERT(false);
        return false;
    }

    if (!mxIsComplex(mx_array))
    {
        std::cerr << "Data is not complex" << std::endl;
        ASSERT(false);
        return false;
    }

    size_t element_count = mxGetNumberOfElements(mx_array);
    const mwSize* rf_size = mxGetDimensions(mx_array);
    dims->x = (uint)rf_size[0];
    dims->y = (uint)rf_size[1];
    dims->z = (uint)rf_size[2];

    cuComplex* data_array_ptr = (cuComplex*)mxGetComplexSingles(mx_array);

    *data_array = new std::vector<cuComplex>(data_array_ptr, &(data_array_ptr[element_count]));

    mxDestroyArray(mx_array);
    matClose(file);

    return true;
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
        matError er = matGetErrno(file_p);
        std::cerr << "Failed to save array to file. Error: " << er << std::endl;
        return false;
    }

    matClose(file_p);

    return true;
}