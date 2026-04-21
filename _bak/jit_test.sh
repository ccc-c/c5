#!/bin/bash
cd /Users/Shared/ccc/project/c5

echo "=== Build elf_m3jit ==="
gcc -w -g -O0 elf_m3jit.c -o elf_m3jit 2>&1 || exit 1
echo "OK"

echo ""
echo "=== Build c5 compiler ==="
gcc -w -g -O1 c5.c c5_elf.c -o c5 2>&1 || exit 1
echo "OK"

echo ""
echo "=== Build c5tool (VM) ==="
gcc -w -g -O1 c5tool.c -o c5tool 2>&1 || exit 1
echo "OK"

echo ""
echo "=== Compile test files ==="
./c5 -o test/hello.elf test/hello.c || exit 1
./c5 -o test/fib.elf test/fib.c || exit 1
./c5 -o test/sum_for.elf test/sum_for.c || exit 1
echo "All compiled"

echo ""
echo "=== Test hello.elf ==="
echo "--- c5tool run ---"
./c5tool run test/hello.elf
echo "--- elf_m3jit ---"
./elf_m3jit test/hello.elf 2>&1 | grep -v "^M3:"

echo ""
echo "=== Test fib.elf ==="
echo "--- c5tool run ---"
./c5tool run test/fib.elf
echo "--- elf_m3jit (debug) ---"
./elf_m3jit test/fib.elf 2>&1 | tail -5

echo ""
echo "=== Test sum_for.elf ==="
echo "--- c5tool run ---"
./c5tool run test/sum_for.elf
echo "--- elf_m3jit (debug) ---"
./elf_m3jit test/sum_for.elf 2>&1 | tail -5

echo ""
echo "=== Done ==="