#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "cuda_transfer.h"


int main()
{
    // Example dimensions and parameters
    int rf_dim[2] = {128, 128}; // Example: 128x128 input
    int output_dim[3] = {64, 64, 64}; // Example: 64x64x64 output
    int n_elements = rf_dim[0] * rf_dim[1];
    int n_output = output_dim[0] * output_dim[1] * output_dim[2] * 2; // 2 for interleaved complex output

    // Allocate input data (int16_t)
    int16_t* rf_data = (int16_t*)malloc(n_elements * sizeof(int16_t));
    if (!rf_data) {
        printf("Failed to allocate rf_data\n");
        return 1;
    }
    // Fill with dummy data
    for (int i = 0; i < n_elements; ++i) rf_data[i] = (int16_t)(i % 32768);

    // Allocate output buffer (interleaved complex values)
    float* output = (float*)malloc(n_output * sizeof(float));
    if (!output) {
        printf("Failed to allocate output\n");
        free(rf_data);
        return 1;
    }
    memset(output, 0, n_output * sizeof(float));

    // Set up parameters
    CudaBeamformerParameters params;
    memset(&params, 0, sizeof(params));
    params.rf_raw_dim[0] = rf_dim[0];
    params.rf_raw_dim[1] = rf_dim[1];
    params.output_points[0] = output_dim[0];
    params.output_points[1] = output_dim[1];
    params.output_points[2] = output_dim[2];
    params.data_type = TYPE_I16; // Set data type to int16_t
    // Set other params as needed...

    // Call the function
    beamform_f32((float*)rf_data, params, output);

    printf("beamform_f32 returned\n");

    // Cleanup
    free(rf_data);
    free(output);

    return 0;
}