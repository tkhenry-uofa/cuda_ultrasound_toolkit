#include <Windows.h>

#include "matlab_transfer.h"



static void*
os_open_shared_memory_area(char* name, size cap)
{
	iptr h = CreateFileMappingA(-1, 0, PAGE_READWRITE, 0, cap, name);
	if (h == INVALID_FILE)
		return NULL;

	return MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, cap);
}


static Pipe
os_open_named_pipe(char* name)
{
	iptr h = CreateNamedPipeA(name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE, 1,
		0, 1 * MEGABYTE, 0, 0);
	return (Pipe) { .file = h, .name = name };
}

static BOOL os_poll_pipe(Pipe p)
{
	DWORD bytes_available = 0;
	return PeekNamedPipe(p.file, 0, 1 * MEGABYTE, 0, &bytes_available, 0) && bytes_available;
}

static ptrdiff_t os_read_pipe(iptr pipe, void* buf, size len)
{
	DWORD total_read = 0;
	ReadFile(pipe, buf, len, &total_read, 0);
	return total_read;
}

