set -x
gcc -w -g -O0 elf_m3jit.c -o elf_m3jit 
./elf_m3jit test/hello.elf