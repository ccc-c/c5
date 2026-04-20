set -x
gcc -w -g -O0 elf_x86jit.c -o elf_x86jit 
./elf_x86jit test/hello.elf