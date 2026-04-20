set -x
./c5 -o test/hello.elf test/hello.c
./c5 -o test/fib.elf test/fib.c
gcc -w -g -O0 elf_x86jit.c -o elf_x86jit
./elf_x86jit test/hello.elf
./elf_x86jit test/fib.elf