#include "TransferServer.h"


TransferServer::TransferServer( const char* command_pipe_name,
                                const char* data_smem_name,
                                const char* header_smem_name )
    : _command_pipe_name( command_pipe_name )
    , _data_smem_name( data_smem_name )
    , _params_smem_name( header_smem_name )
{

    _command_pipe_h = CreateNamedPipeA( _command_pipe_name, PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_NOWAIT,
                                        1, 0, MEGABYTE, 0, 0 );

    if( _command_pipe_h == INVALID_HANDLE_VALUE )
    {
        DWORD error = GetLastError();
        std::cerr << "Error creating command pipe: " << ERROR_MSG(error) << std::endl;
        throw std::runtime_error( "Failed to create command pipe" );
    }

    _data_smem_h = CreateFileMappingA( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, DATA_SMEM_SIZE, _data_smem_name );
    if( _data_smem_h == NULL )  
    {
        DWORD error = GetLastError();
        std::cerr << "Error creating data shared memory: " << ERROR_MSG(error) << std::endl;
        throw std::runtime_error( "Failed to create data shared memory" );
    }

    _data_smem = static_cast< char* >( MapViewOfFile( _data_smem_h, FILE_MAP_ALL_ACCESS, 0, 0, DATA_SMEM_SIZE ) );
    if( _data_smem == NULL )
    {
        DWORD error = GetLastError();
        std::cerr << "Error mapping data shared memory: " << ERROR_MSG(error) << std::endl;
        throw std::runtime_error( "Failed to map data shared memory" );
    }

    _params_smem_h = CreateFileMappingA( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                         0, sizeof( SharedMemoryHeader ), _params_smem_name );
    if( _params_smem_h == NULL )
    {
        DWORD error = GetLastError();
        std::cerr << "Error creating header shared memory: " << ERROR_MSG(error) << std::endl;
        throw std::runtime_error( "Failed to create header shared memory" );
    }
    _parameters_smem = static_cast< SharedMemoryHeader* >( MapViewOfFile( _params_smem_h, FILE_MAP_ALL_ACCESS,
                                                                          0, 0, sizeof( SharedMemoryHeader ) ) );

}

TransferServer::~TransferServer()
{
    if( _data_smem )
    {
        UnmapViewOfFile( _data_smem );
        _data_smem = NULL;
    }

    if( _parameters_smem )
    {
        UnmapViewOfFile( _parameters_smem );
        _parameters_smem = nullptr;
    }

    if( _data_smem_h )
    {
        CloseHandle( _data_smem_h );
        _data_smem_h = nullptr;
    }

    if( _params_smem_h )
    {
        CloseHandle( _params_smem_h );
        _params_smem_h = nullptr;
    }

    if( _command_pipe_h )
    {
        CloseHandle( _command_pipe_h );
        _command_pipe_h = nullptr;
    }
}