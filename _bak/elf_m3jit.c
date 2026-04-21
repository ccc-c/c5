// elf5_jit_arm64.c - AArch64 JIT executor for c5-generated ELF files
//
// Target: Apple Silicon (M1/M2/M3) macOS, AArch64/ARM64
// Usage:  elf5_jit_arm64 [-d] <program.elf> [args...]
//
// Compile on macOS/Apple Silicon:
//   cc -O2 -o elf5_jit_arm64 elf5_jit_arm64.c
//
// Reads a .elf produced by c5 + c5tool, JIT-compiles the c5 VM bytecode
// to native AArch64 machine code in two passes, then runs it directly.
//
// AArch64 register allocation in JIT'd code (AAPCS64 callee-saved = x19-x28):
//   x19 = VM accumulator (a)
//   x20 = VM stack pointer (grows downward, 8-byte longs)
//   x21 = VM data segment base pointer (d_base)
//   x22 = VM frame pointer (bp)
//   x23 = jitmap base pointer  (for LEV indirect return)
//   x24 = VM pool size         (for to_addr bounds check)
//   x9, x10, x11 = scratch (caller-saved, clobbered within snippets)
//   x8  = scratch for C function pointer before BLR
//
// Apple Silicon JIT notes:
//   - mmap with MAP_JIT required for writable+executable memory
//   - pthread_jit_write_protect_np(0/1) to toggle W^X protection
//   - __builtin___clear_cache() to flush instruction cache after emit
//
// Design notes (same as x86-64 version):
//   - text[0] is always padding; compile loop starts at pc=1.
//   - 2-word instructions stamp jitmap[pc+1] = jitmap[pc].
//   - PRTF nargs: text[pc+2] (text[pc+1] is the ADJ opcode value).
//   - Comparisons: pop left into x9, CMP x9,x19, CSET x19,cond.
//   - C functions returning int are sign-extended: SXTW x0,w0.
//   - MOV_imm64 always emits 4 instructions (MOVZ + 3×MOVK) so JSR
//     snippets have a fixed size and pass-2 patching offsets are stable.
//
// Bugs to watch for (from x86-64 experience):
//   1. Compile loop starts at pc=1 (text[0] is padding).
//   2. PRTF nargs is text[pc+2], NOT text[pc+1].
//   3. ENT/LEV must use the correct register (x22 = bp).
//   4. Comparison result register must not overlap operand registers.
//   5. C int-return functions need SXTW sign extension.
//   6. Native stack (SP) must be 16-byte aligned before every BLR.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#ifdef __APPLE__
#include <pthread.h>
#include <libkern/OSCacheControl.h>
#endif

// ---- c5 VM opcodes (must match c5.c / c5tool.c) ----
enum {
  LLA,IMM,JMP,JSR,BZ,BNZ,ENT,ADJ,LEV,LI,LC,SI,SC,PSH,
  OR,XOR,AND,EQ,NE,LT,GT,LE,GE,SHL,SHR,ADD,SUB,MUL,DIV,MOD,
  LF,SF,IMMF,ITF,ITFS,FTI,FADD,FSUB,FMUL,FDIV,
  FEQ,FNE,FLT,FGT,FLE,FGE,PRTF_DBL,
  OPEN,READ,WRIT,CLOS,PRTF,MALC,FREE_OP,MSET,MCMP,EXIT
};

// ---- globals ----
static int      g_debug;
static long     g_poolsz;
static long    *g_text;
static char    *g_data;
static long     g_tlen;
static uint32_t **jitmap;   // jitmap[vm_pc] = native instruction pointer
static uint32_t  *je;       // JIT emit cursor

// ---- ELF helpers ----
static long r16(char *b,long p){return(unsigned char)b[p]|((unsigned char)b[p+1]<<8);}
static long r32(char *b,long p){return r16(b,p)|(r16(b,p+2)<<16);}
static long r64(char *b,long p){
  return(long)((unsigned long)(unsigned int)r32(b,p)|
               ((unsigned long)(unsigned int)r32(b,p+4)<<32));
}

// ---- Emit one 32-bit AArch64 instruction ----
static void emit(uint32_t insn){ *je++ = insn; }

// ---- Register aliases ----
#define VA      19  // VM accumulator
#define VSP     20  // VM stack pointer
#define VDB     21  // VM data base
#define VBP     22  // VM frame pointer (bp)
#define VJMAP   23  // jitmap base
#define VPSZ    24  // VM pool size
#define S0       9  // scratch 0
#define S1      10  // scratch 1
#define S2      11  // scratch 2
#define CFNREG   8  // C function pointer register
#define XZR     31  // zero register
#define SP      31  // stack pointer (same encoding as XZR in non-SP contexts)

// ---- AArch64 instruction encoders ----

// MOV_imm64: always 4 instructions (MOVZ + 3 MOVK) for stable snippet sizes
static void MOV_imm64(int Rd, long v){
  unsigned long u = (unsigned long)v;
  emit(0xD2800000 | (0<<21) | ((u      & 0xFFFF)<<5) | Rd); // MOVZ hw=0
  emit(0xF2800000 | (1<<21) | (((u>>16)& 0xFFFF)<<5) | Rd); // MOVK hw=1
  emit(0xF2800000 | (2<<21) | (((u>>32)& 0xFFFF)<<5) | Rd); // MOVK hw=2
  emit(0xF2800000 | (3<<21) | (((u>>48)& 0xFFFF)<<5) | Rd); // MOVK hw=3
}
// MOV register
static void MOV(int Rd, int Rn){ emit(0xAA0003E0|(Rn<<16)|Rd); }

// Arithmetic (immediate, imm12 ≤ 4095)
static void ADD_imm(int Rd,int Rn,int i){ emit(0x91000000|(i<<10)|(Rn<<5)|Rd); }
static void SUB_imm(int Rd,int Rn,int i){ emit(0xD1000000|(i<<10)|(Rn<<5)|Rd); }
// Arithmetic (register)
static void ADD_reg(int Rd,int Rn,int Rm){ emit(0x8B000000|(Rm<<16)|(Rn<<5)|Rd); }
static void SUB_reg(int Rd,int Rn,int Rm){ emit(0xCB000000|(Rm<<16)|(Rn<<5)|Rd); }
static void MUL_r(int Rd,int Rn,int Rm){ emit(0x9B007C00|(Rm<<16)|(Rn<<5)|Rd); }
static void SDIV   (int Rd,int Rn,int Rm){ emit(0x9AC00C00|(Rm<<16)|(Rn<<5)|Rd); }
// MSUB Rd = Ra - Rn*Rm  (for MOD)
static void MSUB(int Rd,int Rn,int Rm,int Ra){ emit(0x9B008000|(Rm<<16)|(Ra<<10)|(Rn<<5)|Rd); }
// Logic
static void AND_reg(int Rd,int Rn,int Rm){ emit(0x8A000000|(Rm<<16)|(Rn<<5)|Rd); }
static void ORR_reg(int Rd,int Rn,int Rm){ emit(0xAA000000|(Rm<<16)|(Rn<<5)|Rd); }
static void EOR_reg(int Rd,int Rn,int Rm){ emit(0xCA000000|(Rm<<16)|(Rn<<5)|Rd); }
// Shifts
static void LSLV(int Rd,int Rn,int Rm){ emit(0x9AC02000|(Rm<<16)|(Rn<<5)|Rd); }
static void ASRV(int Rd,int Rn,int Rm){ emit(0x9AC02800|(Rm<<16)|(Rn<<5)|Rd); }

// Load/Store (unsigned offset, byte_off must be multiple of 8, range 0..32760)
static void LDR(int Rt,int Rn,int o){ emit(0xF9400000|((o/8)<<10)|(Rn<<5)|Rt); }
static void STR(int Rt,int Rn,int o){ emit(0xF9000000|((o/8)<<10)|(Rn<<5)|Rt); }
// Load byte (zero-extend to 64-bit)
static void LDRB(int Rt,int Rn){ emit(0x39400000|(Rn<<5)|Rt); }
static void STRB(int Rt,int Rn){ emit(0x39000000|(Rn<<5)|Rt); }
// LDR Xd, [Xbase, Xidx, LSL #3]  (pointer-array lookup)
static void LDR_lsl3(int Rt,int Rbase,int Ridx){ emit(0xF8607800|(Ridx<<16)|(Rbase<<5)|Rt); }

// Compare / branch
static void CMP(int Rn,int Rm){  emit(0xEB00001F|(Rm<<16)|(Rn<<5)); }
static void CMP_imm(int Rn,int i){emit(0xF100001F|(i<<10)|(Rn<<5)); }
// CSET Rd, cond  (= CSINC Rd, XZR, XZR, ~cond)
static void CSET(int Rd,int cond){ emit(0x9A9F07E0|((cond^1)<<12)|Rd); }
// Condition codes
#define CC_EQ  0
#define CC_NE  1
#define CC_GE  10
#define CC_LT  11
#define CC_GT  12
#define CC_LE  13
// SXTW: sign-extend 32-bit -> 64-bit
static void SXTW(int Xd,int Wn){ emit(0x93407C00|(Wn<<5)|Xd); }
// Branches
static void B  (int off){ emit(0x14000000|(off&0x3FFFFFF)); }
static void BL (int off){ emit(0x94000000|(off&0x3FFFFFF)); }
static void BR (int Rn) { emit(0xD61F0000|(Rn<<5)); }
static void BLR(int Rn) { emit(0xD63F0000|(Rn<<5)); }
static void RET(void)   { emit(0xD65F03C0); }
static void CBZ (int Rn,int off){ emit(0xB4000000|((off&0x7FFFF)<<5)|Rn); }
static void CBNZ(int Rn,int off){ emit(0xB5000000|((off&0x7FFFF)<<5)|Rn); }
static void Bcond(int c,int off){ emit(0x54000000|((off&0x7FFFF)<<5)|c); }

// Floating-point (double-precision scalar)
static void FMOV_g2f(int Dd,int Xn){ emit(0x9E670000|(Xn<<5)|Dd); } // GP->FP
static void FMOV_f2g(int Xd,int Dn){ emit(0x9E660000|(Dn<<5)|Xd); } // FP->GP
static void FADD_d(int Dd,int Dn,int Dm){ emit(0x1E602800|(Dm<<16)|(Dn<<5)|Dd); }
static void FSUB_d(int Dd,int Dn,int Dm){ emit(0x1E603800|(Dm<<16)|(Dn<<5)|Dd); }
static void FMUL_d(int Dd,int Dn,int Dm){ emit(0x1E600800|(Dm<<16)|(Dn<<5)|Dd); }
static void FDIV_d(int Dd,int Dn,int Dm){ emit(0x1E601800|(Dm<<16)|(Dn<<5)|Dd); }
static void FCMP_d(int Dn,int Dm)       { emit(0x1E602000|(Dm<<16)|(Dn<<5)); }
static void SCVTF (int Dd,int Xn)       { emit(0x9E620000|(Xn<<5)|Dd); }  // int->double
static void FCVTZS(int Xd,int Dn)       { emit(0x9E780000|(Dn<<5)|Xd); }  // double->int(trunc)
static void LDR_d (int Dt,int Rn)       { emit(0xFD400000|(Rn<<5)|Dt); }  // load double
static void STR_d (int Dt,int Rn)       { emit(0xFD000000|(Rn<<5)|Dt); }  // store double

// ---- VM stack helpers ----
// vm_push: *--VSP = VA  (SUB x20,x20,#8; STR x19,[x20])
static void vm_push(void){ SUB_imm(VSP,VSP,8); STR(VA,VSP,0); }
// vm_pop -> S0  (LDR x9,[x20]; ADD x20,x20,#8)
static void vm_pop(void) { LDR(S0,VSP,0); ADD_imm(VSP,VSP,8); }
// vm_peek n -> S0
static void vm_peek(int n){ LDR(S0,VSP,n*8); }

// ---- to_addr: if 0 <= x0 < poolsz: x0 += x21 ----
// Always 5 instructions (fixed size).
static void to_addr_x0(void){
  CMP_imm(0,0);          // CMP x0, #0
  Bcond(CC_LT,4);        // B.LT skip (+4 insns)
  CMP(0,VPSZ);           // CMP x0, x24
  Bcond(CC_GE,2);        // B.GE skip (+2 insns)
  ADD_reg(0,0,VDB);      // ADD x0, x0, x21
  // skip:
}
static void to_addr_s0(void){ MOV(0,S0); to_addr_x0(); } // S0->x0, apply to_addr

// ---- Call a C function: MOV_imm64 x8, fn; BLR x8 ----
// Native SP is always 16-aligned when JIT snippets execute
// (entry wrapper allocated the frame and never modifies SP further).
// BLR x8 pushes ret-addr to LR (x30), not SP — so no alignment concern.
static void call_fn(void *fn){ MOV_imm64(CFNREG,(long)fn); BLR(CFNREG); }

// ---- printf shim ----
static long prtf_shim(long *vsp,long nargs,char *db,long psz){
  long *t=vsp+nargs; long fr=t[-1];
  char *fmt=(fr>=0&&fr<psz)?db+fr:(char*)fr;
  long fa[8]; for(int k=0;k<8;k++) fa[k]=(k<nargs-1)?t[-2-k]:0;
  char *f=fmt; long ai=0,a=0;
  while(*f){
    if(*f!='%'){putchar(*f++);a++;continue;}
    char sp[64],*s=sp;*s++=*f++;
    while(*f=='-'||*f=='+'||*f==' '||*f=='#'||*f=='0')*s++=*f++;
    while(*f>='0'&&*f<='9')*s++=*f++;
    if(*f=='.'){*s++=*f++;while(*f>='0'&&*f<='9')*s++=*f++;}
    if(*f=='l'||*f=='h'||*f=='L')*s++=*f++;
    if(*f=='l')*s++=*f++;
    long cv=*f?*f++:0;*s++=(char)cv;*s=0;
    long v=(ai<8)?fa[ai]:0;
    if(cv=='f'||cv=='g'||cv=='e'||cv=='F'||cv=='G'||cv=='E'){
      double d;memcpy(&d,&v,8);a+=printf(sp,d);
    }else if(cv=='%'){a+=printf("%%");
    }else if(cv=='s'){char *sv=(v>=0&&v<psz)?db+v:(char*)v;a+=printf(sp,sv);
    }else{a+=printf(sp,v);}
    if(cv!='%')ai++;
  }
  return a;
}

// ---- Compile one VM instruction ----
static long compile_one(long pc){
  long op=g_text[pc];
  jitmap[pc]=je;
  if((op>=LLA&&op<=ADJ)||op==IMMF) if(pc+1<g_tlen) jitmap[pc+1]=je;

  if(g_debug){
    static const char *NM[]={"LLA","IMM","JMP","JSR","BZ","BNZ","ENT","ADJ","LEV",
      "LI","LC","SI","SC","PSH","OR","XOR","AND","EQ","NE","LT","GT","LE","GE","SHL",
      "SHR","ADD","SUB","MUL","DIV","MOD","LF","SF","IMMF","ITF","ITFS","FTI","FADD",
      "FSUB","FMUL","FDIV","FEQ","FNE","FLT","FGT","FLE","FGE","PRTF_DBL","OPEN",
      "READ","WRIT","CLOS","PRTF","MALC","FREE","MSET","MCMP","EXIT"};
    if(op>=0&&op<=EXIT){
      fprintf(stderr,"[%4ld] %s",pc,NM[op]);
      if((op>=LLA&&op<=ADJ)||op==IMMF) fprintf(stderr," %ld",g_text[pc+1]);
      fputc('\n',stderr);
    }
  }

  if(op==LLA){
    // x19 = x22 + n*8  (address of local variable n)
    long n=g_text[++pc]; long off=n*8;
    if(off>=0&&off<=4095)       ADD_imm(VA,VBP,(int)off);
    else if(off<0&&off>=-4096)  SUB_imm(VA,VBP,(int)(-off));
    else { MOV_imm64(S0,off); ADD_reg(VA,VBP,S0); }
  }
  else if(op==IMM){ MOV_imm64(VA,g_text[++pc]); }
  else if(op==JMP){ pc++; B(0); }  // 1 insn, patched pass 2
  else if(op==JSR){
    // Fixed layout: MOV_imm64(4) + vm_push(2) + B(1) = 7 insns; B at nat[6]
    long callee=g_text[++pc]; (void)callee;
    MOV_imm64(VA,pc+1);  // push return-pc-index (4 insns)
    vm_push();           // (2 insns)
    B(0);                // jump to callee, patched pass 2 (1 insn)
  }
  else if(op==BZ){  pc++; CBZ(VA,0); }   // 1 insn, patched pass 2
  else if(op==BNZ){ pc++; CBNZ(VA,0); }  // 1 insn, patched pass 2
  else if(op==ENT){
    long n=g_text[++pc];
    SUB_imm(VSP,VSP,8);  STR(VBP,VSP,0);  // *--sp = bp
    MOV(VBP,VSP);                          // bp = sp
    if(n>0){
      long sz=n*8;
      if(sz<=4095) SUB_imm(VSP,VSP,(int)sz);
      else { MOV_imm64(S0,sz); SUB_reg(VSP,VSP,S0); }
    }
  }
  else if(op==ADJ){
    long n=g_text[++pc]; long sz=n*8;
    if(sz>0){
      if(sz<=4095) ADD_imm(VSP,VSP,(int)sz);
      else { MOV_imm64(S0,sz); ADD_reg(VSP,VSP,S0); }
    } else if(sz<0){
      long isz=-sz;
      if(isz<=4095) SUB_imm(VSP,VSP,(int)isz);
      else { MOV_imm64(S0,isz); SUB_reg(VSP,VSP,S0); }
    }
  }
  else if(op==LEV){
    // sp=bp; bp=*sp++; ret_idx=*sp++; jump jitmap[ret_idx]
    MOV(VSP,VBP);
    LDR(VBP,VSP,0); ADD_imm(VSP,VSP,8);  // pop bp
    LDR(S0, VSP,0); ADD_imm(VSP,VSP,8);  // pop ret_idx -> S0
    LDR_lsl3(S1,VJMAP,S0);               // x10 = jitmap[S0]
    BR(S1);                               // jump
  }
  else if(op==LI){
    MOV(0,VA); to_addr_x0(); LDR(VA,0,0);
  }
  else if(op==LC){
    MOV(0,VA); to_addr_x0(); LDRB(VA,0);
  }
  else if(op==SI){
    vm_pop(); to_addr_s0();   // x0 = resolved addr
    STR(VA,0,0);
  }
  else if(op==SC){
    vm_pop(); to_addr_s0();
    STRB(VA,0);
    emit(0x92401E73u);  // AND x19, x19, #0xFF (zero-extend byte)
  }
  else if(op==PSH){ vm_push(); }

  // ---- Integer ALU ----
  else if(op==OR) { vm_pop(); ORR_reg(VA,VA,S0); }
  else if(op==XOR){ vm_pop(); EOR_reg(VA,VA,S0); }
  else if(op==AND){ vm_pop(); AND_reg(VA,VA,S0); }
  else if(op==ADD){ vm_pop(); ADD_reg(VA,S0,VA); }
  else if(op==SUB){ vm_pop(); SUB_reg(VA,S0,VA); }  // S0-VA (left-right)
  else if(op==MUL){ vm_pop(); MUL_r(VA,S0,VA); }
  else if(op==DIV){ vm_pop(); SDIV(VA,S0,VA); }
  else if(op==MOD){
    vm_pop();              // S0=left, VA=right
    SDIV(S1,S0,VA);        // S1 = left/right
    MSUB(VA,S1,VA,S0);     // VA = S0 - S1*VA = left - (left/right)*right
  }
  else if(op==SHL){ vm_pop(); LSLV(VA,S0,VA); }
  else if(op==SHR){ vm_pop(); ASRV(VA,S0,VA); }

  // ---- Comparisons ----
  // S0 = left (*sp++), VA = right (a); CMP S0,VA tests left-right
  else if(op>=EQ&&op<=GE){
    static const int CC[]={CC_EQ,CC_NE,CC_LT,CC_GT,CC_LE,CC_GE};
    vm_pop();
    CMP(S0,VA);
    CSET(VA,CC[op-EQ]);
  }

  // ---- Floating point ----
  // Double bits travel as raw int64 in x19 (same convention as x86-64 version)
  else if(op==IMMF){ MOV_imm64(VA,g_text[++pc]); }
  else if(op==LF){
    // x19 = raw 8 bytes at to_addr(x19)
    MOV(0,VA); to_addr_x0(); LDR(VA,0,0);
  }
  else if(op==SF){
    // *(double*)to_addr(*sp++) = x19  (store raw 8 bytes)
    vm_pop(); to_addr_s0(); STR(VA,0,0);
  }
  else if(op==ITF){
    // integer x19 -> double bits in x19
    SCVTF(0,VA);       // d0 = (double)x19
    FMOV_f2g(VA,0);    // x19 = raw bits of d0
  }
  else if(op==ITFS){
    // *sp (integer) -> double bits at *sp  (in-place)
    LDR(S0,VSP,0);
    SCVTF(0,S0);       // d0 = (double)S0
    FMOV_f2g(S0,0);    // S0 = raw bits
    STR(S0,VSP,0);
  }
  else if(op==FTI){
    FMOV_g2f(0,VA);    // d0 = raw bits as double
    FCVTZS(VA,0);      // x19 = (long)(double)x19  (truncate)
  }
  else if(op==FADD||op==FSUB||op==FMUL||op==FDIV){
    vm_pop();
    FMOV_g2f(0,S0);    // d0 = left
    FMOV_g2f(1,VA);    // d1 = right
    if(op==FADD) FADD_d(0,0,1);
    else if(op==FSUB) FSUB_d(0,0,1);
    else if(op==FMUL) FMUL_d(0,0,1);
    else              FDIV_d(0,0,1);
    FMOV_f2g(VA,0);
  }
  else if(op>=FEQ&&op<=FGE){
    // AArch64 FCMP sets flags like integer CMP for ordered comparisons.
    // Use same condition codes as integer EQ/NE/LT/GT/LE/GE.
    static const int FCC[]={CC_EQ,CC_NE,CC_LT,CC_GT,CC_LE,CC_GE};
    vm_pop();
    FMOV_g2f(0,S0);    // d0 = left
    FMOV_g2f(1,VA);    // d1 = right
    FCMP_d(0,1);       // flags: d0 - d1
    CSET(VA,FCC[op-FEQ]);
  }

  // ---- System calls ----
  else if(op==OPEN){
    // open(to_addr(sp[2]), sp[1], sp[0])
    vm_peek(2); to_addr_s0(); MOV(S2,0);  // S2 = path
    vm_peek(1); MOV(S1,S0);               // S1 = flags
    vm_peek(0);                            // S0 = mode
    MOV(0,S2); MOV(1,S1); MOV(2,S0);
    call_fn((void*)open); SXTW(0,0); MOV(VA,0);
  }
  else if(op==READ){
    // read(sp[2], to_addr(sp[1]), sp[0])
    vm_peek(2); MOV(S2,S0);               // S2 = fd
    vm_peek(1); to_addr_s0(); MOV(S1,0); // S1 = buf
    vm_peek(0);                            // S0 = count
    MOV(0,S2); MOV(1,S1); MOV(2,S0);
    call_fn((void*)read); SXTW(0,0); MOV(VA,0);
  }
  else if(op==WRIT){
    vm_peek(2); MOV(S2,S0);
    vm_peek(1); to_addr_s0(); MOV(S1,0);
    vm_peek(0);
    MOV(0,S2); MOV(1,S1); MOV(2,S0);
    call_fn((void*)write); SXTW(0,0); MOV(VA,0);
  }
  else if(op==CLOS){
    vm_peek(0); MOV(0,S0);
    call_fn((void*)close); SXTW(0,0); MOV(VA,0);
  }
  else if(op==PRTF){
    long nargs=(pc+2<g_tlen&&g_text[pc+1]==ADJ)?g_text[pc+2]:(pc+1<g_tlen?g_text[pc+1]:1);
    MOV(0,VSP);               // x0 = vm_sp
    MOV_imm64(1,nargs);       // x1 = nargs
    MOV(2,VDB);               // x2 = d_base
    MOV_imm64(3,g_poolsz);    // x3 = poolsz
    call_fn((void*)prtf_shim);
    MOV(VA,0);
  }
  else if(op==MALC){
    vm_peek(0); MOV(0,S0);
    call_fn((void*)malloc); MOV(VA,0);
  }
  else if(op==FREE_OP){
    vm_peek(0); MOV(0,S0);
    call_fn((void*)free);
  }
  else if(op==MSET){
    vm_peek(2); to_addr_s0(); MOV(S2,0); // S2 = dst
    vm_peek(1); MOV(S1,S0);              // S1 = val
    vm_peek(0);                           // S0 = count
    MOV(0,S2); MOV(1,S1); MOV(2,S0);
    call_fn((void*)memset); MOV(VA,0);
  }
  else if(op==MCMP){
    vm_peek(2); to_addr_s0(); MOV(S2,0); // S2 = ptr1
    vm_peek(1); to_addr_s0(); MOV(S1,0); // S1 = ptr2
    vm_peek(0);                           // S0 = count
    MOV(0,S2); MOV(1,S1); MOV(2,S0);
    call_fn((void*)memcmp); SXTW(0,0); MOV(VA,0);
  }
  else if(op==EXIT){
    // EXIT: x0 = *sp (exit code), then jump to shared epilogue
    // Fixed 3-instruction snippet: LDR S0,[VSP]; MOV x0,S0; B epilogue(patched)
    vm_peek(0); MOV(0,S0); B(0); // B patched in post-pass
  }
  else if(op==PRTF_DBL){ /* placeholder, unused */ }
  else{
    fprintf(stderr,"JIT ARM64: unknown opcode %ld at vm_pc=%ld\n",op,pc);
    emit(0xD4200000u);  // BRK #0
  }
  return pc;
}

int main(int argc,char **argv){
  int ao=1;
  if(argc<2){fprintf(stderr,"Usage: elf5_jit_arm64 [-d] <prog.elf> [args...]\n");return 1;}
  if(!strcmp(argv[1],"-d")){g_debug=1;ao=2;}
  if(ao>=argc){fprintf(stderr,"Usage: elf5_jit_arm64 [-d] <prog.elf> [args...]\n");return 1;}

  // ---- 1. Read ELF ----
  int fd=open(argv[ao],O_RDONLY);
  if(fd<0){perror(argv[ao]);return 1;}
  g_poolsz=4*1024*1024;
  char *fb=(char*)malloc(g_poolsz);
  long fsz=read(fd,fb,g_poolsz-1); close(fd);
  if(fsz<64){fprintf(stderr,"file too small\n");return 1;}
  if((unsigned char)fb[0]!=0x7f||fb[1]!='E'||fb[2]!='L'||fb[3]!='F'){
    fprintf(stderr,"not ELF\n");return 1;}

  // ---- 2. Parse sections ----
  long eshoff=r64(fb,40),eshnum=r16(fb,60),eshstrndx=r16(fb,62);
  char *shstr=fb+r64(fb,eshoff+eshstrndx*64+24);
  long toff=0,tsz=0,doff=0,dsz=0,soff=0,ssz=0,stroff=0;
  for(long si=0;si<eshnum;si++){
    long b=eshoff+si*64;
    long shn=r32(fb,b),sho=r64(fb,b+24),shs=r64(fb,b+32),shlk=r32(fb,b+40);
    char *n=shstr+shn;
    if(!strcmp(n,".text")){toff=sho;tsz=shs;}
    else if(!strcmp(n,".data")){doff=sho;dsz=shs;}
    else if(!strcmp(n,".symtab")){soff=sho;ssz=shs;stroff=r64(fb,eshoff+shlk*64+24);}
  }
  if(!tsz){fprintf(stderr,"no .text\n");return 1;}

  // ---- 3. Load text/data ----
  g_tlen=tsz/sizeof(long);
  g_text=(long*)calloc(g_tlen+4,sizeof(long));
  g_data=(char*)calloc(dsz+16,1);
  memcpy(g_text,fb+toff,tsz);
  if(dsz>0) memcpy(g_data,fb+doff,dsz);

  // ---- 4. Find main() ----
  long main_off=-1;
  for(long i=0;i<ssz/24;i++){
    long p=soff+i*24;
    if(!strcmp(fb+stroff+r32(fb,p),"main")){main_off=r64(fb,p+8);break;}
  }
  if(main_off<0){fprintf(stderr,"'main' not found\n");return 1;}
  if(g_debug) fprintf(stderr,"ELF: text=%ld longs, data=%ld bytes, main@%ld\n",
                      g_tlen,dsz,main_off);

  // ---- 5. Allocate RWX JIT memory ----
  long jmem_sz=64*1024*1024;
  int mmap_flags = MAP_PRIVATE|MAP_ANON;
#ifdef MAP_JIT
  mmap_flags |= MAP_JIT;  // Required on Apple Silicon for W+X memory
#endif
  void *jitmem=mmap(NULL,jmem_sz,PROT_READ|PROT_WRITE|PROT_EXEC,mmap_flags,-1,0);
  if(jitmem==MAP_FAILED){perror("mmap");return 1;}

  jitmap=(uint32_t**)calloc(g_tlen+4,sizeof(uint32_t*));
  je=(uint32_t*)jitmem;

  // On Apple Silicon: disable W^X write protection before writing JIT code
#if defined(__APPLE__) && defined(__aarch64__)
  pthread_jit_write_protect_np(0);
#endif

  // ---- 6. Emit entry wrapper ----
  // Signature: long jit_entry(long *vm_sp, char *d_base, long *unused)
  //            x0=vm_sp, x1=d_base, x2=unused
  uint32_t *entry = je;

  // Prologue: save callee-saved registers x19-x26, x29, x30 (10 regs = 80 bytes)
  emit(0xA9BB53F3u);  // STP x19, x20, [sp, #-80]!
  emit(0xA9015BF5u);  // STP x21, x22, [sp, #16]
  emit(0xA90263F7u);  // STP x23, x24, [sp, #32]
  emit(0xA9036BF9u);  // STP x25, x26, [sp, #48]
  emit(0xA9047BFDu);  // STP x29, x30, [sp, #64]
  // Set up frame pointer (x29 = sp for ABI compliance)
  MOV(29, SP);

  // Load VM registers
  MOV(VSP, 0);              // x20 = vm_sp
  MOV(VDB, 1);              // x21 = d_base
  MOV(VBP, VSP);            // x22 = initial bp = initial sp
  MOV_imm64(VA, 0);         // x19 = 0  (accumulator)
  MOV_imm64(VPSZ, g_poolsz);// x24 = poolsz

  // jitmap base (x23) is loaded after pass 1 finishes
  uint32_t *jitmap_patch_site = je;
  MOV_imm64(VJMAP, 0);      // x23 = jitmap ptr  (4 insns, patched after pass 1)

  // Jump to main's native code (patched after pass 1)
  uint32_t *entry_jmp = je;
  B(0);                     // B <main>  (patched)

  // ---- Shared epilogue ----
  // Called by EXIT snippets: x0 holds exit code.
  uint32_t *epilogue = je;
  emit(0xA9447BFDu);  // LDP x29, x30, [sp, #64]
  emit(0xA9436BF9u);  // LDP x25, x26, [sp, #48]
  emit(0xA94263F7u);  // LDP x23, x24, [sp, #32]
  emit(0xA9415BF5u);  // LDP x21, x22, [sp, #16]
  emit(0xA8C553F3u);  // LDP x19, x20, [sp], #80
  RET();

  // ---- 7. Pass 1: compile VM instructions (start at pc=1, skip padding) ----
  jitmap[0] = je;
  emit(0xD4200000u);  // BRK #0  (padding slot — should never execute)

  for(long pc=1; pc<g_tlen;){
    pc = compile_one(pc) + 1;
  }
  // Fallthrough sentinel (should not be reached)
  MOV_imm64(0, 0);
  B((int)(epilogue - je));

  // ---- Patch EXIT branch targets ----
  // EXIT snippet layout: LDR S0,[VSP](1) + MOV x0,S0(1) + B(0)(1) = 3 insns; B at nat+2
  for(long pc=1; pc<g_tlen;){
    long op=g_text[pc];
    if(op==EXIT){
      uint32_t *b_insn = jitmap[pc] + 2;
      int rel = (int)(epilogue - b_insn);
      *b_insn = 0x14000000u | (rel & 0x3FFFFFF);
    }
    if((op>=LLA&&op<=ADJ)||op==IMMF) pc+=2; else pc++;
  }

  // ---- 8. Patch jitmap base into entry wrapper ----
  {
    uint32_t *save=je; je=jitmap_patch_site;
    MOV_imm64(VJMAP,(long)(void*)jitmap);
    je=save;
  }

  // ---- 9. Pass 2: patch branch targets ----
  for(long pc=1; pc<g_tlen;){
    long op=g_text[pc];
    if(op<0||op>EXIT){pc++;continue;}
    uint32_t *nat=jitmap[pc];

    if(op==JMP){
      long tgt=g_text[pc+1];
      int rel=(int)(jitmap[tgt]-nat);
      nat[0]=0x14000000u|(rel&0x3FFFFFF);
      pc+=2;
    }
    else if(op==JSR){
      long tgt=g_text[pc+1];
      // JSR: MOV_imm64(4) + vm_push(2) + B(1) = 7 insns; B at nat[6]
      int rel=(int)(jitmap[tgt]-(nat+6));
      nat[6]=0x14000000u|(rel&0x3FFFFFF);
      pc+=2;
    }
    else if(op==BZ){
      long tgt=g_text[pc+1];
      int rel=(int)(jitmap[tgt]-nat);
      nat[0]=0xB4000000u|((rel&0x7FFFF)<<5)|VA;
      pc+=2;
    }
    else if(op==BNZ){
      long tgt=g_text[pc+1];
      int rel=(int)(jitmap[tgt]-nat);
      nat[0]=0xB5000000u|((rel&0x7FFFF)<<5)|VA;
      pc+=2;
    }
    else if((op>=LLA&&op<=ADJ)||op==IMMF){pc+=2;}
    else{pc++;}
  }

  // Patch entry B to main
  {
    int rel=(int)(jitmap[main_off]-entry_jmp);
    *entry_jmp=0x14000000u|(rel&0x3FFFFFF);
  }

  if(g_debug) fprintf(stderr,"JIT ARM64: %ld instructions emitted\n",
                      (long)(je-(uint32_t*)jitmem));

  // Re-enable W^X protection and flush instruction cache
#if defined(__APPLE__) && defined(__aarch64__)
  pthread_jit_write_protect_np(1);
  sys_icache_invalidate(jitmem, (char*)je-(char*)jitmem);
#endif

  // ---- 10. Set up VM stack and run ----
  long vm_stk_sz=1024*1024;
  long *vm_stk=(long*)malloc(vm_stk_sz);
  long *vsp=(long*)((char*)vm_stk+vm_stk_sz);

  // Push initial frame: argc, argv, sentinel return address
  long exit_idx=g_tlen-2;
  *--vsp=(long)(argc-ao);
  *--vsp=(long)(argv+ao);
  *--vsp=exit_idx;

  typedef long(*jfn_t)(long*,char*,long*);
  return (int)((jfn_t)(void*)entry)(vsp, g_data, g_text);
}