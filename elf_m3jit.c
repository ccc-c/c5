// elf_m3jit.c - ARM64 JIT compiler for c5 ELF
// For Apple Silicon (M1/M2/M3) Macs

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

enum { LLA=0 ,IMM=1 ,JMP=2 ,JSR=3 ,BZ=4 ,BNZ=5 ,ENT=6 ,ADJ=7 ,LEV=8 ,LI=9 ,LC=10 ,SI=11 ,SC=12 ,PSH=13 ,
OR=14 ,XOR=15 ,AND=16 ,EQ=17 ,NE=18 ,LT=19 ,GT=20 ,LE=21 ,GE=22 ,SHL=23 ,SHR=24 ,ADD=25 ,SUB=26 ,MUL=27 ,DIV=28 ,MOD=29 ,
LF=30 ,SF=31 ,IMMF=32,ITF=33,ITFS=34,FTI=35 ,FADD=36,FSUB=37,FMUL=38,FDIV=39,
FEQ=40 ,FNE=41 ,FLT=42 ,FGT=43 ,FLE=44 ,FGE=45 ,PRTF_DBL=46,
OPEN=47,READ=48,WRIT=49,CLOS=50,PRTF=51,MALC=52,FREE=53,MSET=54,MCMP=55,EXIT=56 };

enum { RELA_DATA = 1, RELA_FUNC = 2, RELA_JMP = 3 };

long poolsz;
long debug;

long r16(char *buf, long p) { return (buf[p] & 0xFF) | ((buf[p+1] & 0xFF) << 8); }
long r32(char *buf, long p) { return r16(buf,p) | (r16(buf,p+2) << 16); }
long r64(char *buf, long p) { return r32(buf,p) | (r32(buf,p+4) << 32); }
void mem_cpy(char *dst, char *src, long n) { long i=0; while (i<n) { dst[i]=src[i]; i=i+1; } }
double long_to_double(long v) { return *(double *)&v; }
long double_to_long(double d) { return *(long *)&d; }

char *jitmem;
char *je;
char **jitmap;

long emit_off; // offset for second pass patching

// ARM64 registers
enum { X0=0, X1=1, X2=2, X3=3, X4=4, X5=5, X6=6, X7=7, X8=8, X9=9, X10=10, X11=11, X12=12, X13=13, X14=14, X15=15,
       X16=16, X17=17, X18=18, X19=19, X20=20, X21=21, X22=22, X23=23, X24=24, X25=25, X26=26, X27=27, X28=28, X29=29, X30=30, SP=31 };

long re(long r) { return r & 0x1f; }

void emit32(long val) {
    *(long *)je = val;
    je += 4;
}

void patch32(long pos, long val) {
    *(long *)(jitmem + pos) = val;
}

// str xd, [xb, #off]
void emit_str(long xd, long xb, long off) {
    long imm9 = off & 0x1ff;
    emit32(0xF8000000 | (re(xd) << 0) | (re(xb) << 5) | (imm9 << 12));
}

// ldr xd, [xb, #off]
void emit_ldr(long xd, long xb, long off) {
    long imm9 = off & 0x1ff;
    emit32(0xF8400000 | (re(xd) << 0) | (re(xb) << 5) | (imm9 << 12));
}

void emit_str_sp(long xd, long off) { emit_str(xd, SP, off); }
void emit_ldr_sp(long xd, long off) { emit_ldr(xd, SP, off); }

// ldrb wd, [xb, #off]
void emit_ldrb(long xd, long xb, long off) {
    emit32(0x39400000 | (re(xd) << 0) | (re(xb) << 5) | ((off & 0x1ff) << 12));
}

// strb wd, [xb, #off]
void emit_strb(long xd, long xb, long off) {
    emit32(0x39000000 | (re(xd) << 0) | (re(xb) << 5) | ((off & 0x1ff) << 12));
}

void emit_movz(long xd, long imm) {
    long shift = 0;
    if (imm > 0xffff) { shift = 16; imm >>= 16; }
    if (imm > 0xffff) { shift = 32; imm >>= 16; }
    if (imm > 0xffff) { shift = 48; imm >>= 16; }
    emit32(0xD2800000 | (re(xd) << 0) | ((imm & 0xffff) << 5) | ((shift/16) << 21));
}

void emit_mov(long xd, long xs) {
    emit32(0xAA200000 | (re(xd) << 0) | (re(31) << 5) | (re(xs) << 16));
}

void emit_add_imm(long xd, long xn, long imm) {
    emit32(0x91000000 | (re(xd) << 0) | (re(xn) << 5) | ((imm & 0xfff) << 10));
}

void emit_add(long xd, long xn, long xm) {
    emit32(0x8B000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_sub_imm(long xd, long xn, long imm) {
    emit32(0xD1000000 | (re(xd) << 0) | (re(xn) << 5) | ((imm & 0xfff) << 10));
}

void emit_sub(long xd, long xn, long xm) {
    emit32(0xCB000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_subs(long xd, long xn, long xm) {
    emit32(0xEB000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_mul(long xd, long xn, long xm) {
    emit32(0x9B000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_sdiv(long xd, long xn, long xm) {
    emit32(0x9AC00C00 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_and(long xd, long xn, long xm) {
    emit32(0x8A000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_orr(long xd, long xn, long xm) {
    emit32(0xAA000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_eor(long xd, long xn, long xm) {
    emit32(0xCA000000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_lsl(long xd, long xn, long xm) {
    emit32(0x9AC02000 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_lsr(long xd, long xn, long xm) {
    emit32(0x9AC02400 | (re(xd) << 0) | (re(xn) << 5) | (re(xm) << 16));
}

void emit_sxtb(long xd, long xn) {
    emit32(0x93C00000 | (re(xd) << 0) | (re(xn) << 5));
}

// cset xd, cond
void emit_cset(long xd, long cond) {
    emit32(0x9A9F0000 | (re(xd) << 0) | (((cond ^ 1)) << 5));
}

void emit_push(long xreg) {
    emit_sub_imm(SP, SP, 8);
    emit_str(xreg, SP, 0);
}

void emit_pop(long xreg) {
    emit_ldr(xreg, SP, 0);
    emit_add_imm(SP, SP, 8);
}

void emit_ret() {
    emit32(0xD65F0000 | (30 << 5));
}

void emit_nop() {
    emit32(0xD503201F);
}

// b offset
void emit_b(long off) {
    long adj = (off - 4) >> 2;
    emit32(0x14000000 | (adj & 0x3FFFFFF));
}

// bl offset
void emit_bl(long off) {
    long adj = (off - 4) >> 2;
    emit32(0x94000000 | (adj & 0x3FFFFFF));
}

// cbz xd, offset
void emit_cbz(long xd, long off) {
    long adj = (off - 4) >> 2;
    emit32(0x34000000 | (re(xd) << 0) | ((adj & 0x7FFF) << 5));
}

// cbnz xd, offset
void emit_cbnz(long xd, long off) {
    long adj = (off - 4) >> 2;
    emit32(0x35000000 | (re(xd) << 0) | ((adj & 0x7FFF) << 5));
}

// blr xn
void emit_blr(long xn) {
    emit32(0xD63F0000 | (re(xn) << 5));
}

// br xn
void emit_br(long xn) {
    emit32(0xD61F0000 | (re(xn) << 5));
}

void emit_prtf() {
    printf("JIT: printf called\n");
}

long load_and_jit(char *filename, long vm_argc, char **vm_argv);

long load_elf_and_run(char *filename, long argc, char **argv) {
    long fd;
    char *file_buf;
    long file_size;
    long e_shoff, e_shnum;
    long sh_text, sh_data, sh_symtab, sh_strtab, sh_rela;
    long text_off, text_sz, data_off, data_sz;
    long symtab_off, symtab_sz, strtab_off, strtab_sz, rela_off, rela_sz;
    long i, num_syms, sym_p, name_off;
    char *sym_name;
    long main_offset;
    long text_count;
    long (*jitmain)(long, char**);
    
    poolsz = 1024 * 1024;
    
    fd = open(filename, 0, 0);
    if (fd < 0) { printf("Failed to open %s\n", filename); return -1; }
    
    file_buf = (char *)malloc(poolsz);
    file_size = read(fd, file_buf, poolsz);
    close(fd);
    
    if (file_buf[0] != 0x7f || file_buf[1] != 'E' || file_buf[2] != 'L' || file_buf[3] != 'F') {
        printf("Error: Not a valid ELF file.\n");
        return -1;
    }
    
    e_shoff = r64(file_buf, 40);
    e_shnum = r16(file_buf, 60);
    
    sh_text = e_shoff + 64 * 1;
    sh_data = e_shoff + 64 * 2;
    sh_symtab = e_shoff + 64 * 3;
    sh_strtab = e_shoff + 64 * 4;
    sh_rela = e_shoff + 64 * 6;
    
    text_off = r64(file_buf, sh_text + 24);
    text_sz = r64(file_buf, sh_text + 32);
    
    data_off = r64(file_buf, sh_data + 24);
    data_sz = r64(file_buf, sh_data + 32);
    
    symtab_off = r64(file_buf, sh_symtab + 24);
    symtab_sz = r64(file_buf, sh_symtab + 32);
    
    strtab_off = r64(file_buf, sh_strtab + 24);
    strtab_sz = r64(file_buf, sh_strtab + 32);
    
    rela_off = r64(file_buf, sh_rela + 24);
    rela_sz = r64(file_buf, sh_rela + 32);
    
    long *text_base = (long *)malloc(text_sz + 8);
    char *data_base = (char *)malloc(data_sz + 8);
    
    mem_cpy((char *)text_base, file_buf + text_off, text_sz);
    mem_cpy(data_base, file_buf + data_off, data_sz);
    
    text_count = text_sz / sizeof(long);
    
    // Find main symbol
    main_offset = -1;
    num_syms = symtab_sz / 24;
    i = 0;
    while (i < num_syms) {
        sym_p = symtab_off + i * 24;
        name_off = r32(file_buf, sym_p + 0);
        sym_name = file_buf + strtab_off + name_off;
        
        if (sym_name[0] == 'm' && sym_name[1] == 'a' && sym_name[2] == 'i' && sym_name[3] == 'n' && sym_name[4] == 0) {
            main_offset = r64(file_buf, sym_p + 8);
            break;
        }
        i = i + 1;
    }
    
    if (main_offset == -1) {
        printf("Error: 'main' symbol not found in ELF!\n");
        return -1;
    }
    
    // Apply relocations
    long *linked_text = text_base;
    long j = 0;
    while (j < rela_sz / 24) {
        long r_offset, r_symidx, r_type;
        long target;
        
        r_offset = r64(file_buf, rela_off + j * 24 + 0);
        r_symidx = r64(file_buf, rela_off + j * 24 + 8);
        r_type = r64(file_buf, rela_off + j * 24 + 16);
        
        target = r_offset;
        
        if (r_symidx == -1) {
            if (r_type == RELA_FUNC || r_type == RELA_JMP) {
                // text slot index - no change needed for single ELF
            } else if (r_type == RELA_DATA) {
                linked_text[target] = linked_text[target] + (long)data_base;
            }
        }
        j = j + 1;
    }
    
    // Allocate executable memory for JIT
    jitmem = mmap(0, poolsz, PROT_EXEC | PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (!jitmem) {
        printf("could not mmap jit executable memory\n");
        return -1;
    }
    
    jitmap = (char **)(jitmem + poolsz / 2);
    
    je = jitmem;
    
    // Track branch info for second pass
    long branch_info[1024][2]; // [slot][0=offset in jitmem, 1=target slot]
    long branch_count = 0;
    
    // ===== PASS 1: Emit ARM64 code =====
    long pc_idx = 0;
    while (pc_idx <= text_count) {
        long i = text_base[pc_idx];
        
        jitmap[pc_idx] = je; // record position
        
        if (debug) {
            printf("%04x: slot=%ld opcode=%ld", (long)(je - jitmem), pc_idx, i);
            if (i <= ADJ || i == IMMF) printf(" operand=%ld", text_base[pc_idx+1]);
            printf("\n");
        }
        
        if (i == LLA) {
            long off = text_base[pc_idx + 1];
            emit_add_imm(X0, SP, off * 4);
            pc_idx += 2;
        }
        else if (i == IMM) {
            long val = text_base[pc_idx + 1];
            emit_movz(X0, val);
            pc_idx += 2;
        }
        else if (i == PSH) {
            emit_push(X0);
            pc_idx++;
        }
        else if (i == LI) {
            emit_ldr(X0, X0, 0);
            pc_idx++;
        }
        else if (i == LC) {
            emit_ldrb(X0, X0, 0);
            emit_sxtb(X0, X0);
            pc_idx++;
        }
        else if (i == SI) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_str(X0, X1, 0);
            pc_idx++;
        }
        else if (i == SC) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_strb(X0, X1, 0);
            pc_idx++;
        }
        else if (i == ADD) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_add(X0, X1, X0);
            pc_idx++;
        }
        else if (i == SUB) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_sub(X0, X1, X0);
            pc_idx++;
        }
        else if (i == MUL) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_mul(X0, X1, X0);
            pc_idx++;
        }
        else if (i == DIV) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_sdiv(X0, X1, X0);
            pc_idx++;
        }
        else if (i == MOD) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_sdiv(X2, X1, X0);
            emit_mul(X2, X2, X0);
            emit_sub(X0, X1, X2);
            pc_idx++;
        }
        else if (i == AND) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_and(X0, X1, X0);
            pc_idx++;
        }
        else if (i == OR) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_orr(X0, X1, X0);
            pc_idx++;
        }
        else if (i == XOR) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_eor(X0, X1, X0);
            pc_idx++;
        }
        else if (i == SHL) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_lsl(X0, X1, X0);
            pc_idx++;
        }
        else if (i == SHR) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_lsr(X0, X1, X0);
            pc_idx++;
        }
        else if (i == EQ || i == NE || i == LT || i == GT || i == LE || i == GE) {
            emit_ldr(X1, SP, 0);
            emit_add_imm(SP, SP, 8);
            emit_subs(X0, X1, X0);
            if (i == EQ) emit_cset(X0, 0);
            else if (i == NE) emit_cset(X0, 1);
            else if (i == LT) emit_cset(X0, 3);
            else if (i == GT) emit_cset(X0, 4);
            else if (i == LE) emit_cset(X0, 5);
            else if (i == GE) emit_cset(X0, 6);
            pc_idx++;
        }
        else if (i == ENT) {
            long framesize = text_base[pc_idx + 1];
            emit_sub_imm(SP, SP, 16);
            emit_str(X29, SP, 0);
            emit_str(X30, SP, 8);
            emit_mov(X29, SP);
            if (framesize > 0) emit_sub_imm(SP, SP, framesize * 4);
            pc_idx += 2;
        }
        else if (i == ADJ) {
            long off = text_base[pc_idx + 1];
            if (off > 0) emit_add_imm(SP, SP, off * 4);
            pc_idx += 2;
        }
        else if (i == LEV) {
            emit_mov(SP, X29);
            emit_ldr(X29, SP, 0);
            emit_ldr(X30, SP, 8);
            emit_add_imm(SP, SP, 16);
            emit_ret();
            pc_idx++;
        }
        else if (i == JMP) {
            long target = text_base[pc_idx + 1];
            branch_info[branch_count][0] = je - jitmem;
            branch_info[branch_count][1] = target;
            emit_b(0); // placeholder
            branch_count++;
            pc_idx += 2;
        }
        else if (i == JSR) {
            long target = text_base[pc_idx + 1];
            branch_info[branch_count][0] = je - jitmem;
            branch_info[branch_count][1] = target;
            emit_bl(0); // placeholder
            branch_count++;
            pc_idx += 2;
        }
        else if (i == BZ) {
            long target = text_base[pc_idx + 1];
            branch_info[branch_count][0] = je - jitmem;
            branch_info[branch_count][1] = target;
            emit_cbz(X0, 0); // placeholder
            branch_count++;
            pc_idx += 2;
        }
        else if (i == BNZ) {
            long target = text_base[pc_idx + 1];
            branch_info[branch_count][0] = je - jitmem;
            branch_info[branch_count][1] = target;
            emit_cbnz(X0, 0); // placeholder
            branch_count++;
            pc_idx += 2;
        }
        else if (i >= OPEN && i <= EXIT) {
            pc_idx++; // skip ADJ count
            if (i == EXIT) {
                // Emit code to call native exit
                emit_movz(X0, 0); // return 0 for now
                emit_bl((long)exit - (long)(je + 4)); // bl exit
            }
            pc_idx++;
        }
        else {
            printf("JIT: unknown opcode %ld at slot %ld\n", i, pc_idx);
            return -1;
        }
    }
    
    // Make sure we have room for the exit code
    long exit_code_pos = je - jitmem;
    
    // ===== PASS 2: Fixup branches =====
    long k = 0;
    while (k < branch_count) {
        long pos = branch_info[k][0];
        long target = branch_info[k][1];
        long target_pos = jitmap[target] - jitmem;
        long offset = target_pos - pos;
        
        // Patch the branch instruction
        long instr = *(long *)(jitmem + pos);
        long new_instr;
        
        if ((instr & 0xFC000000) == 0x14000000) {
            // b instruction
            long adj = (offset - 4) >> 2;
            new_instr = 0x14000000 | (adj & 0x3FFFFFF);
        } else if ((instr & 0xFC000000) == 0x94000000) {
            // bl instruction
            long adj = (offset - 4) >> 2;
            new_instr = 0x94000000 | (adj & 0x3FFFFFF);
        } else if ((instr & 0xFC000000) == 0x34000000) {
            // cbz instruction
            long adj = (offset - 4) >> 2;
            new_instr = 0x34000000 | (instr & (0x1F << 0)) | ((adj & 0x7FFF) << 5);
        } else if ((instr & 0xFC000000) == 0x35000000) {
            // cbnz instruction
            long adj = (offset - 4) >> 2;
            new_instr = 0x35000000 | (instr & (0x1F << 0)) | ((adj & 0x7FFF) << 5);
        } else {
            printf("JIT: unknown branch instruction at pos %ld: 0x%lx\n", pos, instr);
            new_instr = instr; // keep as-is
        }
        
        patch32(pos, new_instr);
        k++;
    }
    
    long entry_offset = jitmap[main_offset] - jitmem;
    
    printf("JIT: compiled %ld bytecode slots to ARM64\n", text_count);
    printf("JIT: code size = %ld bytes\n", je - jitmem);
    printf("JIT: entry at offset 0x%lx\n", entry_offset);
    
    // Call the jitted code
    jitmain = (void *)(jitmem + entry_offset);
    return jitmain(vm_argc, vm_argv);
}

int main(int argc, char **argv) {
    long result;
    
    if (argc < 2) {
        printf("Usage: elf_m3jit [-d] <program.elf> [args...]\n");
        return -1;
    }
    
    debug = 0;
    if (argv[1][0] == '-' && argv[1][1] == 'd') {
        debug = 1;
        argc--; argv++;
    }
    
    result = load_elf_and_run(argv[1], argc - 1, &argv[1]);
    
    return (int)result;
}