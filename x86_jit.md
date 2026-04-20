guest@localhost:~/c5$ ./x86_jit.sh
++ ./c5 -o test/hello.elf test/hello.c
>> ELF written to test/hello.elf (syms=1, relas=1) <<
++ ./c5 -o test/fib.elf test/fib.c
>> ELF written to test/fib.elf (syms=2, relas=6) <<
++ gcc -w -g -O0 elf_x86jit.c -o elf_x86jit
++ ./elf_x86jit test/hello.elf
hello, world
++ ./elf_x86jit test/fib.elf
f(7)=13