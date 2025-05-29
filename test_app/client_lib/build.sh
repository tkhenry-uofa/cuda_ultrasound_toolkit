

cuda_lib_header_path="../../include/cuda_toolkit"
header_paths="-I ../ -I ${cuda_lib_header_path}"

output_dir="output"
mkdir -p ${output_dir}

header_file="./cuda_transfer.h"
output_header="${output_dir}/cuda_transfer_matlab.h"

cflags="-march=native -std=c11 -Wall ${header_paths}"
debug_flags="${cflags} -O0 -fuse-ld=lld -g -gcodeview -Wl,--pdb="
libcflags="${cflags} -fPIC -shared -Wno-unused-variable "
ldflags="${ldflags} -lm -lgdi32 -lwinmm -L./${output_dir}"

matlab_libcflags="${libcflags} -DMATLAB_CONSOLE"
matlab_ldflags="-llibmat -llibmex"

cc=${CC:-cc}
build=release

${cc} ${libcflags} ${debug_flags} cuda_transfer.c -o ${output_dir}/cuda_transfer_debug.dll ${ldflags}
${cc} ${matlab_libcflags} cuda_transfer.c -o ${output_dir}/cuda_transfer_matlab.dll -L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' ${matlab_ldflags}
${cc} ${cflags} ${debug_flags} test_lib.c -o ${output_dir}/test_lib.exe ${ldflags} -lcuda_transfer_debug



cp -f ${header_file} ${output_header}
cp -f ../parameter_defs.h ${output_dir}/parameter_defs.h
cp -f ${cuda_lib_header_path}/cuda_beamformer_parameters.h ${output_dir}/cuda_beamformer_parameters.h