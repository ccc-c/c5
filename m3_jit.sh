set -x
./c5 -o test/hello.elf test/hello.c
./c5 -o test/fib.elf test/fib.c
./c5 -o test/float.elf test/float.c
gcc -w -fsanitize=address -g -O1 -fno-omit-frame-pointer elf_m3jit.c -o elf_m3jit
./elf_m3jit test/hello.elf
./elf_m3jit test/fib.elf
./elf_m3jit test/float.elf