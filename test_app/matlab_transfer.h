#ifndef MATLAB_TRANSFER_H
#define MATLAB_TRANSFER_H

#include "defs.h"


namespace matlab_transfer
{

	bool _nack_response();

	uint write_to_pipe(Handle pipe, void* data, uint len);

	void* _open_shared_memory_area(char* name, size cap);

	int _poll_pipe(Handle p);

	uint _read_pipe(Handle pipe, void* buf, size len);


	bool create_resources(BeamformerParametersFull** bp_mem_h, Handle* input_pipe);
	bool wait_for_data(Handle pipe, void** data, uint* bytes, uint timeout = 0);

	Handle open_output_pipe(char* name);

}


#endif // !MATLAB_TRANSFER_H
