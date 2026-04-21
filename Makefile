CC = gcc
CFLAGS = -w -g -O1 -fno-omit-frame-pointer

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# imac (Darwin arm64) -> M3 JIT, Linux x86_64 -> x86 JIT
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    JIT_TARGET = elf_m3jit
  else
    JIT_TARGET = elf_x86jit
  endif
else ifeq ($(UNAME_S),Linux)
  ifeq ($(UNAME_M),x86_64)
    JIT_TARGET = elf_x86jit
  else
    JIT_TARGET = elf_m3jit
  endif
endif

.PHONY: all test test_linker selfhost test_all clean elf_m3jit elf_x86jit

# 預設建置所有 C5 基礎核心執行檔
all: c5 c5tool

# 核心編譯器 (由 c5.c 前端語法解析 + c5_elf.c 後端寫出 ELF 組成)
c5: c5.c c5_elf.c
	$(CC) $(CFLAGS) c5.c c5_elf.c -o c5

# 核心工具鏈 (包含虛擬機 VM 與靜態連結器 Linker)
c5tool: c5tool.c
	$(CC) $(CFLAGS) c5tool.c -o c5tool

# 基礎語法與 VM 測試 (涵蓋原本的 test.sh)
test: all
	@echo "--- 執行基本編譯與 VM 測試 ---"
	./c5 -s test/fib.c || true
	./c5 test/fib.c || true
	./c5 hello.c || true
	@# 測試編譯器遇到無 main 函數的錯誤捕捉 (會預期出錯)
	./c5 c5.c c5.c hello.c || true
	./c5 test/sum_for.c || true
	./c5 test/float.c || true
	./c5 test/init.c || true
	./c5 test/error.c || true
	./c5 test/comment.c || true
	./c5 -o test/fib.elf test/fib.c
	./c5 -o test/lib_counter.elf test/lib_counter.c
	./c5 -o test/main_counter.elf test/main_counter.c
	@echo "--- 基本測試完成 ---"

# 多檔靜態連結器測試 (Linker Test)
test_linker: all
	@echo "--- 執行多檔靜態連結 (Static Linking) 測試 ---"
	./c5 -o test/lib_math.elf test/lib_math.c
	./c5 -o test/main_ext.elf test/main_ext.c
	./c5tool link -o test/run.elf test/lib_math.elf test/main_ext.elf
	./c5tool run test/run.elf || true
	@echo "--- 靜態連結測試完成 ---"

# 終極模組化自我編譯測試 (Self-Hosting Test)
selfhost: all
	@echo "--- 執行多檔模組化的 Self-hosting 自我編譯測試 ---"
	@# 1. 讓 c5 編譯自己的原始碼模組
	./c5 -o c5.elf c5.c
	./c5 -o c5_elf.elf c5_elf.c
	@# 2. 利用靜態連結器，將 C5 前端與 ELF 後端組裝為完整的虛擬化編譯器
	./c5tool link -o c5_final.elf c5.elf c5_elf.elf
	@# 3. 把 c5tool 也虛擬化編譯成 ELF
	./c5 -o c5tool.elf c5tool.c
	@# 4. 確保在虛擬機 (c5tool) 執行的 c5_final.elf，能精準編譯 fib.c
	./c5tool run c5_final.elf -o test/fib2.elf test/fib.c || true
	@# 5. 用虛擬化出的 c5tool 載入並執行剛編譯出來的 fib2.elf
	./c5tool run c5tool.elf run test/fib2.elf || true
	@echo "--- 自我編譯大賽完成！ ---"


# 一鍵執行所有流程
test_all: test test_linker selfhost $(JIT_TARGET)

# 清理編譯產生的執行檔與中介物件檔
clean:
	rm -rf c5 c5tool *.elf test/*.elf *.o *.dSYM elf_m3jit elf_x86jit
	@echo "清理完成！"

# M3 JIT 測試
elf_m3jit: all
	./c5 -o test/hello.elf test/hello.c
	./c5 -o test/fib.elf test/fib.c
	./c5 -o test/float.elf test/float.c
	gcc -w -fsanitize=address -g -O1 -fno-omit-frame-pointer elf_m3jit.c -o elf_m3jit
	./elf_m3jit test/hello.elf || true
	./elf_m3jit test/fib.elf || true
	./elf_m3jit test/float.elf || true

# x86 JIT 測試
elf_x86jit: all
	./c5 -o test/hello.elf test/hello.c
	./c5 -o test/fib.elf test/fib.c
	./c5 -o test/float.elf test/float.c
	gcc -w -g -O0 elf_x86jit.c -o elf_x86jit
	./elf_x86jit test/hello.elf || true
	./elf_x86jit test/fib.elf || true
	./elf_x86jit test/float.elf || true
