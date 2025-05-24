

cflags="-march=native -std=c11 -Wall -I./external/include"
#cflags="${cflags} -fsanitize=address,undefined"
#cflags="${cflags} -fproc-stat-report"
#cflags="${cflags} -Rpass-missed=.*"
libcflags="${cflags} -fPIC -shared -Wno-unused-variable"
ldflags="-lm"

cc=${CC:-cc}
build=release


ldflags="${ldflags} -lgdi32 -lwinmm"

libcflags="${libcflags} -DMATLAB_CONSOLE"
extra_ldflags="-llibmat -llibmex"

${cc} ${libcflags} cuda_transfer.c -o cuda_transfer.dll \
	-L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' \
	${extra_ldflags}
	

