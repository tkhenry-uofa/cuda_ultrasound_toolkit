#ifndef MATLAB_TRANSFER_H
#define MATLAB_TRANSFER_H

#include "defs.h"


namespace matlab_transfer
{

	bool _nack_response();

	size _write_to_pipe(char* name, void* data, size len);

	void* _open_shared_memory_area(char* name, size cap);

	Pipe _open_named_pipe(char* name);

	int _poll_pipe(Pipe p);

	ptrdiff_t _read_pipe(iptr pipe, void* buf, size len);


	bool create_resources(void** bp_mem_h, void** input_pipe);
	bool wait_for_params(BeamformerParameters** bp, void* mem_h);



}


#endif // !MATLAB_TRANSFER_H
