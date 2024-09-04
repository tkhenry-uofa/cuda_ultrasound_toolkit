#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <iostream>
#include "defs.h"

class SparseMatrix
{
public:
   

    // Proper RAII has this throw if something goes wrong, that exception must not be propagated out of the library or will crash
    SparseMatrix(uint value_count, uint square_dim, float* values, int* rows, int* colums)
        :_dimension(square_dim), _d_columns(nullptr), _d_rows(nullptr), _d_values(nullptr), _value_count(value_count)
    {
        size_t data_size = _value_count * sizeof(float);
        CUDA_THROW_IF_ERR(cudaMalloc((void**)&_d_columns, data_size));
        CUDA_THROW_IF_ERR(cudaMalloc((void**)&_d_rows, data_size));
        CUDA_THROW_IF_ERR(cudaMalloc((void**)&_d_values, data_size));

        CUDA_THROW_IF_ERR(cudaMemcpy(_d_columns, colums, data_size, cudaMemcpyDefault));
        CUDA_THROW_IF_ERR(cudaMemcpy(_d_rows, rows, data_size, cudaMemcpyDefault));
        CUDA_THROW_IF_ERR(cudaMemcpy(_d_values, values, data_size, cudaMemcpyDefault));
        
        cusparseStatus_t status = cusparseCreateConstCsr(&descriptor, 
            _dimension, 
            _dimension, 
            _value_count, 
            _d_rows, 
            _d_columns, 
            _d_values, 
            CUSPARSE_INDEX_32I, 
            CUSPARSE_INDEX_32I, 
            CUSPARSE_INDEX_BASE_ZERO, 
            CUDA_R_32F);

        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout << "Failed to create sparse array." << std::endl;
            ASSERT(false);
            throw std::runtime_error("Failed to create sparse array");
        }
    }

    ~SparseMatrix()
    {
        if (descriptor)
        {
            cusparseDestroySpMat(descriptor);
            cudaFree(_d_columns);
            cudaFree(_d_rows);
            cudaFree(_d_values);
        }
    }

    cusparseConstSpMatDescr_t descriptor = nullptr;
//private:

    uint _value_count;
    uint _dimension;
    float* _d_values;
    int* _d_rows;
    int* _d_columns;
};

#endif // SPARSE_MATRIX_H

