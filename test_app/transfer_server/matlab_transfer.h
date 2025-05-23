#ifndef MATLAB_TRANSFER_H
#define MATLAB_TRANSFER_H

#include "../defs.h"


namespace matlab_transfer
{

	bool _nack_response();

	bool write_to_pipe(Handle pipe, void* data, size_t len);

	void* _open_shared_memory_area(const char* name, size cap);

	int _poll_pipe(Handle* p);

	uint _read_pipe(Handle pipe, void* buf, size len);

	bool close_pipe(Handle pipe);

	bool create_input_pipe(Handle* pipe);

	bool create_smem(BeamformerParametersFull** bp_mem_h);
	bool wait_for_data(Handle pipe, void* data, uint* bytes, uint timeout = 0);

	Handle open_output_pipe(const char* name);

	bool disconnect_pipe(Handle pipe);

	int last_error();


}


#endif // !MATLAB_TRANSFER_H
