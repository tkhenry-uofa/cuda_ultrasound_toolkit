#include "TransferServer.h"

#define COMMAND_WAIT_TIMEOUT 60 * 60 * 1000 // 1 hour
#define COMMAND_POLL_PERIOD 100 // 100ms

TransferServer::TransferServer( const char* command_pipe_name,
                                const char* data_smem_name,
                                const char* header_smem_name )
    : _command_pipe_name( command_pipe_name )
    , _data_smem_name( data_smem_name )
    , _params_smem_name( header_smem_name )
{

    //SIZE_T large_page_min = GetLargePageMinimum();

    _command_pipe_h = CreateNamedPipeA( _command_pipe_name, PIPE_ACCESS_DUPLEX, PIPE_TYPE_MESSAGE | PIPE_NOWAIT,
                                        1, 0, MEGABYTE, 0, 0 );

    if( _command_pipe_h == INVALID_HANDLE_VALUE )
    {
        WINDOWS_ERROR_MESSAGE( "Error creating command pipe", GetLastError() );
        throw std::runtime_error( "Failed to create command pipe" );
    }

    _data_smem_h = CreateFileMappingA( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, DATA_SMEM_SIZE, _data_smem_name );
    if( _data_smem_h == NULL )
    {
        WINDOWS_ERROR_MESSAGE( "Error creating data shared memory", GetLastError() );
        throw std::runtime_error( "Failed to create data shared memory" );
    }

    _data_smem = static_cast<char*>(MapViewOfFile( _data_smem_h, FILE_MAP_ALL_ACCESS, 0, 0, DATA_SMEM_SIZE ));
    if( _data_smem == nullptr )
    {
        WINDOWS_ERROR_MESSAGE( "Error mapping data shared memory", GetLastError() );
        throw std::runtime_error( "Failed to map data shared memory" );
    }

    // Test writing to the shared memory
    memset( _data_smem, 1, DATA_SMEM_SIZE );

    _data_smem[0x40000000u] = 0xFFu; // Test writing to the shared memory

    _params_smem_h = CreateFileMappingA( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                         0, sizeof( SharedMemoryParams ), _params_smem_name );
    if( _params_smem_h == NULL )
    {
        WINDOWS_ERROR_MESSAGE( "Error creating header shared memory", GetLastError() );
        throw std::runtime_error( "Failed to create header shared memory" );
    }
    _parameters_smem = static_cast< SharedMemoryParams* >( MapViewOfFile( _params_smem_h, FILE_MAP_ALL_ACCESS,
                                                                          0, 0, sizeof( SharedMemoryParams ) ) );

    if( _parameters_smem == NULL )
    {
        WINDOWS_ERROR_MESSAGE( "Error mapping header shared memory", GetLastError() );
        throw std::runtime_error( "Failed to map header shared memory" );
    }

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

std::optional<CommandPipeMessage> TransferServer::wait_for_command()
{
    DWORD bytes_read = 0;
    CommandPipeMessage command;

    uint elapsed = 0;

    while( elapsed < COMMAND_WAIT_TIMEOUT )
    {
        if( ReadFile( _command_pipe_h, &command, sizeof( command ), &bytes_read, NULL ) )
        {
            if( bytes_read == sizeof( command ) )
            {
                if( command.opcode == CudaCommand::ERR )
                {
                    std::cerr << "Error in command pipe: " << command.opcode << std::endl;
                    respond_error();
                    return std::nullopt;
                }
                if( command.data_size > DATA_SMEM_SIZE )
                {
                    std::cerr << "Data size too large: " << command.data_size << ", Max: " << DATA_SMEM_SIZE << std::endl;
                    respond_error();
                    return std::nullopt;
                }
                return command;
            }
        }

        DWORD error = GetLastError();
        if( error != ERROR_NO_DATA && error != ERROR_MORE_DATA && error != ERROR_PIPE_LISTENING)
        {
            std::cerr << "Error reading from command pipe: " << format_windows_error_message(error) << std::endl;
            return std::nullopt;
        }

        Sleep( COMMAND_POLL_PERIOD );
        elapsed += COMMAND_POLL_PERIOD;
    }

    std::cerr << "Command wait timed out" << std::endl;
    return std::nullopt;
}

bool TransferServer::write_output( const void* data, size_t size )
{
    DWORD bytes_written = 0;
    if( !WriteFile( _command_pipe_h, data, static_cast< DWORD >( size ), &bytes_written, NULL ) )
    {
        DWORD error = GetLastError();
        std::cerr << "Error writing to command pipe: " << format_windows_error_message(error) << std::endl;
        return false;
    }

    return true;
}