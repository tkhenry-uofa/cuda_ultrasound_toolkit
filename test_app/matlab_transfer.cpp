#include <Windows.h>

#include "matlab_transfer.h"



bool
matlab_transfer::_nack_response()
{
	return true;
}

size
matlab_transfer::_write_to_pipe(char* name, void* data, size len)
{
	HANDLE pipe = CreateFileA(name, GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);

	i32 bytes_written;
	WriteFile(pipe, data, len, &bytes_written, 0);
	return bytes_written;
}


void*
matlab_transfer::_open_shared_memory_area(char* name, size cap)
{
	HANDLE h = CreateFileMappingA(-1, 0, PAGE_READWRITE, 0, cap, name);
	if (h == INVALID_FILE)
		return NULL;

	return MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, cap);
}


Pipe
matlab_transfer::_open_named_pipe(char* name)
{
	iptr h = CreateNamedPipeA(name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE, 1,
		0, 1 * MEGABYTE, 0, 0);
	return (Pipe) { .file = h, .name = name };
}

int matlab_transfer::_poll_pipe(Pipe p)
{
	DWORD bytes_available = 0;
	return (int)(PeekNamedPipe(p.file, 0, 1 * MEGABYTE, 0, &bytes_available, 0) && bytes_available);
}

ptrdiff_t matlab_transfer::_read_pipe(iptr pipe, void* buf, size len)
{
	DWORD total_read = 0;
	ReadFile(pipe, buf, len, &total_read, 0);
	return total_read;
}

bool matlab_transfer::create_resources(void** bp_mem_h, void** input_pipe)
{
	return true;
}
bool matlab_transfer::wait_for_params(BeamformerParameters** bp, void* mem_h)
{
	return true;
}


