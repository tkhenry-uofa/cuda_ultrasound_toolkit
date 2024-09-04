#ifndef SPARSE_FILTERS_H
#define SPARSE_FILTERS_H

#include "../sparse_matrix.h"

namespace sparse_filters
{
	SparseMatrix* 
	create_hilbert_filter(uint sample_count)
	{
		int* rows = new int[sample_count];
		int* cols = new int[sample_count];
		float* values = new float[sample_count];

		uint pivot = sample_count / 2;
		for (uint i = 0; i < sample_count; i++)
		{
			rows[i] = cols[i] = i;
			
			if (i == 0 || i == pivot)
			{
				values[i] = 1.0f;
			}
			else if (i > 0 && i < pivot)
			{
				values[i] = 2.0f;
			}
			else
			{
				values[i] = 0.0f;
			}
		}

		SparseMatrix* output = nullptr;
		try
		{
			output = new SparseMatrix(sample_count, sample_count, values, rows, cols);
		}
		catch (std::runtime_error)
		{
			std::cout << "Failed to create hilbert filter matrix." << std::endl;
			ASSERT(false);

			delete[] rows, cols, values;
			return nullptr;
		}

		delete[] rows, cols, values;
		return output;
	}

	SparseMatrix* create_filter(uint filter_length, uint sample_count, float* filter)
	{
		size_t value_count = filter_length * sample_count;
		int* rows = new int[value_count];
		int* cols = new int[value_count];
		float* values = new float[value_count];

		for (uint col_idx = 0; col_idx < sample_count; col_idx++)
		{
			for (uint value_idx = 0; value_idx < filter_length; value_idx++)
			{
				size_t output_idx = col_idx * filter_length + value_idx;

				if (col_idx + value_idx >= sample_count || output_idx >= value_count) continue;

				rows[output_idx] = col_idx + value_idx; // Diagonal matrix
				cols[output_idx] = col_idx;
				values[output_idx] = filter[value_idx];
			}
		}

		SparseMatrix* output = nullptr;
		try
		{
			output = new SparseMatrix(filter_length, sample_count, values, rows, cols);
		}
		catch (std::runtime_error)
		{
			std::cout << "Failed to create filter matrix." << std::endl;
			ASSERT(false);

			delete[] rows, cols, values;
			return nullptr;
		}

		delete[] rows, cols, values;
		return output;
	}
}
#endif // !SPARSE_FILTERS_H