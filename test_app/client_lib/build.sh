

cuda_lib_header_path="../../include/cuda_toolkit"
header_paths="-I ../ -I ${cuda_lib_header_path}"


cflags="-march=native -std=c11 -Wall"
libcflags="${cflags} -fPIC -shared -Wno-unused-variable ${header_paths}"
ldflags="-lm"

output_dir="output"
mkdir -p ${output_dir}


cc=${CC:-cc}
build=release


cflags="${cflags} ${header_paths}"

debug_flags="${cflags} -O0 -fuse-ld=lld -g -gcodeview -Wl,--pdb="	

ldflags="${ldflags} -lm -lgdi32 -lwinmm -L./${output_dir}"

${cc} ${libcflags} ${debug_flags} cuda_transfer_new.c -o ${output_dir}/cuda_transfer_new.dll \
	${ldflags}


matlab_libcflags="${libcflags} -DMATLAB_CONSOLE"
matlab_ldflags="-llibmat -llibmex"

${cc} ${matlab_libcflags} cuda_transfer.c -o ${output_dir}/cuda_transfer.dll \
	-L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' \
	${matlab_ldflags}

${cc} ${matlab_libcflags} cuda_transfer_new.c -o ${output_dir}/cuda_transfer_new_matlab.dll \
	-L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' \
	${matlab_ldflags}

${cc} ${cflags} ${debug_flags} test_lib.c -o ${output_dir}/test_lib.exe ${ldflags} -lcuda_transfer_new

header_file="cuda_transfer_new.h"
output_header="${output_dir}/cuda_transfer_new_matlab.h"

#clang -x c -std=c11 ${header_paths} -E -P -C "${header_file}" > "${output_header}"


cp -f cuda_transfer.h ${output_dir}/cuda_transfer.h
cp -f cuda_transfer_new.h ${output_dir}/cuda_transfer_new.h
cp -f cuda_transfer_new.h ${output_dir}/cuda_transfer_new_matlab.h
cp -f ../parameter_defs.h ${output_dir}/parameter_defs.h

cp -f ../../include/cuda_toolkit/cuda_beamformer_parameters.h ${output_dir}/cuda_beamformer_parameters.h