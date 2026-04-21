// elf5_jit.c - x86-64 JIT executor for c5-generated ELF files
//
// Usage: elf5_jit [-d] <program.elf> [args...]
//
// Reads a .elf produced by c5 + c5tool, JIT-compiles the c5 VM bytecode
// to native x86-64 machine code in two passes, then runs it directly.
//
// Register allocation in JIT code:
//   %rax  = VM accumulator (a)
//   %r12  = VM stack ptr (grows downward, 8-byte longs)
//   %r13  = VM data segment base pointer
//   %r15  = VM frame pointer (bp)
//
// Key design notes:
//   - text[0] is always padding (c5 uses *++e for first emit, leaving e[0] empty)
//   - Compile loop starts at pc=1 to get correct jitmap entries
//   - 2-word instructions (have operand) also stamp jitmap for the operand slot
//     so that any jump targeting it lands at the instruction's native code

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// ---- opcodes (must match c5.c) ----
enum {
  LLA,IMM,JMP,JSR,BZ,BNZ,ENT,ADJ,LEV,LI,LC,SI,SC,PSH,
  OR,XOR,AND,EQ,NE,LT,GT,LE,GE,SHL,SHR,ADD,SUB,MUL,DIV,MOD,
  LF,SF,IMMF,ITF,ITFS,FTI,FADD,FSUB,FMUL,FDIV,
  FEQ,FNE,FLT,FGT,FLE,FGE,PRTF_DBL,
  OPEN,READ,WRIT,CLOS,PRTF,MALC,FREE_OP,MSET,MCMP,EXIT
};

static int    g_debug;
static long   g_poolsz;
static long  *g_text;
static char  *g_data;
static long   g_tlen;

// jitmap[vm_pc] -> native byte address for that VM instruction
static unsigned char **jitmap;
static unsigned char  *je;   // JIT emit cursor

// ---- ELF helpers ----
static long r16(char *b,long p){return(unsigned char)b[p]|((unsigned char)b[p+1]<<8);}
static long r32(char *b,long p){return r16(b,p)|(r16(b,p+2)<<16);}
static long r64(char *b,long p){
  return(long)((unsigned long)(unsigned int)r32(b,p)|
               ((unsigned long)(unsigned int)r32(b,p+4)<<32));
}

// ---- emit helpers ----
static void e1(unsigned char a){*je++=a;}
static void e2(unsigned char a,unsigned char b){*je++=a;*je++=b;}
static void e3(unsigned char a,unsigned char b,unsigned char c){*je++=a;*je++=b;*je++=c;}
static void e4(unsigned char a,unsigned char b,unsigned char c,unsigned char d){
  *je++=a;*je++=b;*je++=c;*je++=d;}
static void ei32(int v){
  *je++=(unsigned char)(v);*je++=(unsigned char)(v>>8);
  *je++=(unsigned char)(v>>16);*je++=(unsigned char)(v>>24);}
static void ei64(long v){
  unsigned long u=(unsigned long)v;
  *je++=u;*je++=u>>8;*je++=u>>16;*je++=u>>24;
  *je++=u>>32;*je++=u>>40;*je++=u>>48;*je++=u>>56;}

// movabs $v, %rax / %r11 / %rsi / %rcx / %rdx / %rdi
static void mrax(long v){e2(0x48,0xB8);ei64(v);}
static void mr11(long v){e2(0x49,0xBB);ei64(v);}
static void mrsi(long v){e2(0x48,0xBE);ei64(v);}
static void mrcx(long v){e2(0x48,0xB9);ei64(v);}
static void mrdi(long v){e2(0x48,0xBF);ei64(v);}

// ---- VM stack ops (r12 = VM sp, 8-byte slots, grows down) ----
// push %rax -> VM stack
static void vpush(void){e4(0x49,0x83,0xEC,0x08);e4(0x49,0x89,0x04,0x24);}
// pop VM stack -> %rcx
static void vpop_rcx(void){e4(0x49,0x8B,0x0C,0x24);e4(0x49,0x83,0xC4,0x08);}
// peek VM sp[n*8] -> %rcx  (n=0,1,2)
static void vpeek(int n){
  if(n==0){e4(0x49,0x8B,0x0C,0x24);}
  else{e1(0x49);e1(0x8B);e1(0x4C);e1(0x24);e1((unsigned char)(n*8));}
}

// ---- to_addr: if 0 <= %rdi < poolsz: %rdi += %r13; else nop ----
// input/output in %rdi; clobbers %rsi as temp
static void toa_rdi(void){
  e3(0x48,0x85,0xFF);            // test %rdi,%rdi
  e2(0x7C,0x0E);                 // jl skip (+14)
  e2(0x48,0xBE);ei64(g_poolsz); // movabs $psz,%rsi
  e3(0x48,0x39,0xF7);            // cmp %rsi,%rdi
  e2(0x7D,0x03);                 // jge skip (+3)
  e3(0x4C,0x01,0xEF);            // add %r13,%rdi
}
// to_addr %rax -> %rax (via %rdi)
static void toa_rax(void){e3(0x48,0x89,0xC7);toa_rdi();e3(0x48,0x89,0xF8);}
// to_addr %rcx -> %rdi
static void toa_rcx_rdi(void){e3(0x48,0x89,0xCF);toa_rdi();}
// to_addr %rcx -> %rsi (for 2nd arg)
static void toa_rcx_rsi(void){toa_rcx_rdi();e3(0x48,0x89,0xFE);}

// ---- call %r11 with 16-byte alignment ----
static void ccall(void){
  // rsp is 8-aligned during JIT execution. push %rbx makes it 16-aligned.
  // CALL then pushes ret-addr, leaving callee with 8-aligned rsp — exactly
  // what System V ABI expects on function entry, so the callee's push %rbp
  // produces a 16-aligned rbp suitable for MOVAPS and other aligned stores.
  e1(0x53);                      // push %rbx  (8-aligned -> 16-aligned)
  e3(0x41,0xFF,0xD3);            // call *%r11 (16-aligned -> 8-aligned at callee entry)
  e1(0x5B);                      // pop  %rbx
}
static void cfn(void *fn){mr11((long)fn);ccall();}

// ---- LEV jump: jmp jitmap[%r11] ----
// movabs $jitmap,%r10; jmp *(%r10,%r11,8)
static void lev_jmp(void){
  e2(0x4D,0xBA);ei64((long)(void*)jitmap);
  e4(0x4B,0xFF,0x24,0xDA); // jmp *(%r10,%r11,8): REX=4B FF /4 SIB(3,r11,r10)=DA
}

// ---- standard epilogue (restore callee-saved regs + ret) ----
static void epilogue(void){
  e1(0x5B);           // pop %rbx
  e2(0x41,0x5F);      // pop %r15
  e2(0x41,0x5E);      // pop %r14
  e2(0x41,0x5D);      // pop %r13
  e2(0x41,0x5C);      // pop %r12
  e1(0x5D);           // pop %rbp
  e1(0xC3);           // ret
}

// ---- prtf shim ----
// Called as prtf_shim(vm_sp, nargs, d_base, poolsz)
static long prtf_shim(long *vsp,long nargs,char *db,long psz){
  long *t=vsp+nargs;
  long fr=t[-1];
  char *fmt=(fr>=0&&fr<psz)?db+fr:(char*)fr;
  long fa[8];
  for(int k=0;k<8;k++) fa[k]=(k<nargs-1)?t[-2-k]:0;
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
    }else if(cv=='s'){
      char *sv=(v>=0&&v<psz)?db+v:(char*)v;a+=printf(sp,sv);
    }else{a+=printf(sp,v);}
    if(cv!='%')ai++;
  }
  return a;
}

// ---- compile one VM instruction at *pc; advance pc; return new pc ----
static long compile_one(long pc){
  long op=g_text[pc];

  // Record jitmap for this pc
  jitmap[pc]=je;
  // For 2-word instructions, also stamp the operand slot so that
  // any function whose entry index falls on the operand gets a valid native ptr.
  if((op>=LLA&&op<=ADJ)||op==IMMF){
    if(pc+1<g_tlen) jitmap[pc+1]=je;
  }

  static const char *NM[]={"LLA","IMM","JMP","JSR","BZ","BNZ","ENT","ADJ","LEV","LI","LC","SI",
    "SC","PSH","OR","XOR","AND","EQ","NE","LT","GT","LE","GE","SHL","SHR","ADD","SUB","MUL",
    "DIV","MOD","LF","SF","IMMF","ITF","ITFS","FTI","FADD","FSUB","FMUL","FDIV",
    "FEQ","FNE","FLT","FGT","FLE","FGE","PRTF_DBL","OPEN","READ","WRIT","CLOS","PRTF",
    "MALC","FREE","MSET","MCMP","EXIT"};
  if(g_debug&&op>=0&&op<=EXIT){
    fprintf(stderr,"[%4ld] %s",pc,NM[op]);
    if((op>=LLA&&op<=ADJ)||op==IMMF) fprintf(stderr," %ld",g_text[pc+1]);
    fputc('\n',stderr);
  }

  if(op==LLA){
    long n=g_text[++pc]; long off=n*8;
    if(off>=-128&&off<=127){e4(0x49,0x8D,0x47,(unsigned char)(off&0xFF));}
    else{e3(0x49,0x8D,0x87);ei32((int)off);}
  }
  else if(op==IMM){ mrax(g_text[++pc]); }
  else if(op==JMP){ pc++; e1(0xE9);ei32(0); }
  else if(op==JSR){
    // push return pc index onto VM stack, then jump to callee
    long callee=g_text[++pc];
    mrax(pc+1); vpush(); (void)callee;
    e1(0xE9);ei32(0);
  }
  else if(op==BZ){  pc++; e3(0x48,0x85,0xC0); e2(0x0F,0x84);ei32(0); }
  else if(op==BNZ){ pc++; e3(0x48,0x85,0xC0); e2(0x0F,0x85);ei32(0); }
  else if(op==ENT){
    long n=g_text[++pc];
    e4(0x49,0x83,0xEC,0x08);       // sub $8,%r12   (push bp)
    e3(0x4D,0x89,0x3C);e1(0x24);  // mov %r15,(%r12)
    e3(0x4D,0x89,0xE7);            // mov %r12,%r15  (bp=sp)
    if(n>0){
      long sz=n*8;
      if(sz<=127) e4(0x49,0x83,0xEC,(unsigned char)(sz&0xFF));
      else{e3(0x49,0x81,0xEC);ei32((int)sz);}
    }
  }
  else if(op==ADJ){
    long n=g_text[++pc]; long sz=n*8;
    if(sz>0){
      if(sz<=127) e4(0x49,0x83,0xC4,(unsigned char)(sz&0xFF));
      else{e3(0x49,0x81,0xC4);ei32((int)sz);}
    } else if(sz<0){
      // negative ADJ (rare): sub from sp
      long isz=-sz;
      if(isz<=127) e4(0x49,0x83,0xEC,(unsigned char)(isz&0xFF));
      else{e3(0x49,0x81,0xEC);ei32((int)isz);}
    }
  }
  else if(op==LEV){
    e3(0x4D,0x89,0xFC);            // mov %r15,%r12  (sp=bp)
    e3(0x4D,0x8B,0x3C);e1(0x24);  // mov (%r12),%r15 (pop bp)
    e4(0x49,0x83,0xC4,0x08);       // add $8,%r12
    e4(0x4D,0x8B,0x1C,0x24);       // mov (%r12),%r11 (pop ret idx)
    e4(0x49,0x83,0xC4,0x08);       // add $8,%r12
    lev_jmp();
  }
  else if(op==LI){ toa_rax(); e3(0x48,0x8B,0x00); }
  else if(op==LC){ toa_rax(); e3(0x0F,0xB6,0x00); }
  else if(op==SI){
    e1(0x50);          // push %rax (native: save value-to-store)
    vpop_rcx();        // rcx = address (from VM stack)
    toa_rcx_rdi();     // rdi = resolved address
    e1(0x59);          // pop %rcx  (rcx = value)
    e3(0x48,0x89,0x0F);// mov %rcx,(%rdi)
  }
  else if(op==SC){
    e1(0x50);
    vpop_rcx(); toa_rcx_rdi(); e1(0x59);
    e2(0x88,0x0F);     // mov %cl,(%rdi)
    e3(0x0F,0xB6,0xC1);// movzbl %cl,%eax  (a = byte written)
  }
  else if(op==PSH){ vpush(); }
  else if(op==OR) { vpop_rcx();e3(0x48,0x09,0xC8); }
  else if(op==XOR){ vpop_rcx();e3(0x48,0x31,0xC8); }
  else if(op==AND){ vpop_rcx();e3(0x48,0x21,0xC8); }
  else if(op==ADD){ vpop_rcx();e3(0x48,0x01,0xC8); }
  else if(op==SUB){ vpop_rcx();e3(0x48,0x29,0xC1);e3(0x48,0x89,0xC8); }
  else if(op==MUL){ vpop_rcx();e4(0x48,0x0F,0xAF,0xC1); }
  else if(op==DIV){
    vpop_rcx();                    // rcx=left, rax=right
    e3(0x48,0x87,0xC1);            // xchg: rax=left, rcx=right
    e2(0x48,0x99);                 // cqo
    e3(0x48,0xF7,0xF9);            // idiv %rcx  -> rax=quot
  }
  else if(op==MOD){
    vpop_rcx();
    e3(0x48,0x87,0xC1);
    e2(0x48,0x99);
    e3(0x48,0xF7,0xF9);            // idiv %rcx  -> rdx=rem
    e3(0x48,0x89,0xD0);            // mov %rdx,%rax
  }
  else if(op==SHL){ vpop_rcx();e3(0x48,0x87,0xC1);e3(0x48,0xD3,0xE0); }
  else if(op==SHR){ vpop_rcx();e3(0x48,0x87,0xC1);e3(0x48,0xD3,0xF8); }
  else if(op>=EQ&&op<=GE){
    unsigned char CC[]={0x94,0x95,0x9C,0x9F,0x9E,0x9D};
    vpop_rcx();
    // cmp %rax,%rcx: flags = rcx - rax (rcx=left/*sp, rax=right/a)
    // Use rdx as scratch so we don't clobber rax (right operand) before cmp
    e3(0x48,0x31,0xD2);            // xor %rdx,%rdx  (zero, doesn't touch flags or rax)
    e3(0x48,0x39,0xC1);            // cmp %rax,%rcx  (flags: rcx-rax, rax intact)
    e3(0x0F,CC[op-EQ],0xD2);       // setXX %dl
    e3(0x48,0x89,0xD0);            // mov %rdx,%rax
  }
  else if(op==IMMF){ mrax(g_text[++pc]); }
  else if(op==LF){
    toa_rax();
    e4(0xF2,0x0F,0x10,0x00);       // movsd (%rax),%xmm0
    e4(0x66,0x48,0x0F,0x7E);e1(0xC0);// movq %xmm0,%rax
  }
  else if(op==SF){
    e1(0x50); vpop_rcx(); toa_rcx_rdi(); e1(0x58); // pop %rax
    e4(0x66,0x48,0x0F,0x6E);e1(0xC0); // movq %rax,%xmm0
    e4(0xF2,0x0F,0x11,0x07);           // movsd %xmm0,(%rdi)
  }
  else if(op==ITF){
    e4(0xF2,0x48,0x0F,0x2A);e1(0xC0); // cvtsi2sdq %rax,%xmm0
    e4(0x66,0x48,0x0F,0x7E);e1(0xC0);
  }
  else if(op==ITFS){
    e4(0x49,0x8B,0x0C,0x24);           // mov (%r12),%rcx
    e4(0xF2,0x48,0x0F,0x2A);e1(0xC1); // cvtsi2sdq %rcx,%xmm0
    e4(0x66,0x48,0x0F,0x7E);e1(0xC1); // movq %xmm0,%rcx
    e4(0x49,0x89,0x0C,0x24);           // mov %rcx,(%r12)
  }
  else if(op==FTI){
    e4(0x66,0x48,0x0F,0x6E);e1(0xC0);
    e4(0xF2,0x48,0x0F,0x2C);e1(0xC0); // cvttsd2si %xmm0,%rax
  }
  else if(op==FADD||op==FSUB||op==FMUL||op==FDIV){
    vpop_rcx();
    e4(0x66,0x48,0x0F,0x6E);e1(0xC1); // movq %rcx,%xmm0 (left)
    e4(0x66,0x48,0x0F,0x6E);e1(0xC8); // movq %rax,%xmm1 (right)
    if(op==FADD)e4(0xF2,0x0F,0x58,0xC1);
    else if(op==FSUB)e4(0xF2,0x0F,0x5C,0xC1);
    else if(op==FMUL)e4(0xF2,0x0F,0x59,0xC1);
    else e4(0xF2,0x0F,0x5E,0xC1);
    e4(0x66,0x48,0x0F,0x7E);e1(0xC0);
  }
  else if(op>=FEQ&&op<=FGE){
    unsigned char FC[]={0x94,0x95,0x92,0x97,0x96,0x93};
    vpop_rcx();
    e4(0x66,0x48,0x0F,0x6E);e1(0xC1);  // movq %rcx,%xmm0 (left)
    e4(0x66,0x48,0x0F,0x6E);e1(0xC8);  // movq %rax,%xmm1 (right)
    e3(0x48,0x31,0xD2);                 // xor %rdx,%rdx (zero result reg, no flag change)
    e4(0x66,0x0F,0x2E,0xC1);            // ucomisd %xmm1,%xmm0 (xmm0 vs xmm1, flags: xmm0-xmm1)
    e3(0x0F,FC[op-FEQ],0xD2);           // setXX %dl
    e3(0x48,0x89,0xD0);                 // mov %rdx,%rax
  }
  // ---- syscalls ----
  else if(op==OPEN){
    vpeek(2); toa_rcx_rdi();          // rdi=path
    e1(0x57);                          // push %rdi (save)
    e1(0x49);e1(0x8B);e1(0x74);e1(0x24);e1(0x08); // mov 8(%r12),%rsi (flags)
    vpeek(0); e3(0x48,0x89,0xCA);     // rdx=mode
    e1(0x5F);                          // pop %rdi
    cfn((void*)open);
  e3(0x48,0x63,0xC0);            // movsxd %eax,%rax (sign-extend int->long)
  }
  else if(op==READ){
    vpeek(2); e3(0x48,0x89,0xCF);     // rdi=fd
    e1(0x57);
    vpeek(1); toa_rcx_rsi();           // rsi=buf
    vpeek(0); e3(0x48,0x89,0xCA);     // rdx=cnt
    e1(0x5F);
    cfn((void*)read);
  e3(0x48,0x63,0xC0);            // movsxd %eax,%rax (sign-extend int->long)
  }
  else if(op==WRIT){
    vpeek(2); e3(0x48,0x89,0xCF);
    e1(0x57);
    vpeek(1); toa_rcx_rsi();
    vpeek(0); e3(0x48,0x89,0xCA);
    e1(0x5F);
    cfn((void*)write);
  e3(0x48,0x63,0xC0);            // movsxd %eax,%rax (sign-extend int->long)
  }
  else if(op==CLOS){
    vpeek(0); e3(0x48,0x89,0xCF); cfn((void*)close);
  e3(0x48,0x63,0xC0);            // movsxd %eax,%rax (sign-extend int->long)
  }
  else if(op==PRTF){
    // nargs is the next ADJ operand (lookahead, NOT consumed here)
    long nargs=(pc+2<g_tlen&&g_text[pc+1]==ADJ)?g_text[pc+2]:((pc+1<g_tlen)?g_text[pc+1]:1);
    e3(0x4C,0x89,0xE7);  // mov %r12,%rdi  (vm_sp)
    mrsi(nargs);
    e3(0x4C,0x89,0xEA);  // mov %r13,%rdx  (d_base)
    mrcx(g_poolsz);
    cfn((void*)prtf_shim);
  }
  else if(op==MALC){ vpeek(0);e3(0x48,0x89,0xCF);cfn((void*)malloc); }
  else if(op==FREE_OP){ vpeek(0);e3(0x48,0x89,0xCF);cfn((void*)free); }
  else if(op==MSET){
    vpeek(2); toa_rcx_rdi();
    e1(0x57);
    vpeek(1); e3(0x48,0x89,0xCE);     // rsi=val
    vpeek(0); e3(0x48,0x89,0xCA);     // rdx=count
    e1(0x5F);
    cfn((void*)memset);
  }
  else if(op==MCMP){
    vpeek(2); toa_rcx_rdi();
    e1(0x57);
    vpeek(1); toa_rcx_rsi();
    vpeek(0); e3(0x48,0x89,0xCA);
    e1(0x5F);
    cfn((void*)memcmp);
  e3(0x48,0x63,0xC0);            // movsxd %eax,%rax (sign-extend int->long)
  }
  else if(op==EXIT){
    vpeek(0); e3(0x48,0x89,0xC8);     // rax = exit code
    epilogue();
  }
  else if(op==PRTF_DBL){
    // PRTF_DBL is a no-op placeholder in c5; skip
  }
  else{
    fprintf(stderr,"JIT: unknown opcode %ld at vm_pc=%ld\n",op,pc);
    e2(0x0F,0x0B); // ud2
  }
  return pc;
}

int main(int argc,char **argv){
  int ao=1;
  if(argc<2){fprintf(stderr,"Usage: elf5_jit [-d] <prog.elf> [args...]\n");return 1;}
  if(!strcmp(argv[1],"-d")){g_debug=1;ao=2;}
  if(ao>=argc){fprintf(stderr,"Usage: elf5_jit [-d] <prog.elf> [args...]\n");return 1;}
  char *ef=argv[ao];

  // ---- read ELF ----
  int fd=open(ef,O_RDONLY);
  if(fd<0){perror(ef);return 1;}
  g_poolsz=4*1024*1024;
  char *fb=(char*)malloc(g_poolsz);
  long fsz=read(fd,fb,g_poolsz-1); close(fd);
  if(fsz<64){fprintf(stderr,"file too small\n");return 1;}
  if((unsigned char)fb[0]!=0x7f||fb[1]!='E'||fb[2]!='L'||fb[3]!='F'){
    fprintf(stderr,"not ELF\n");return 1;}

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

  g_tlen=tsz/sizeof(long);
  g_text=(long*)calloc(g_tlen+4,sizeof(long));
  g_data=(char*)calloc(dsz+16,1);
  memcpy(g_text,fb+toff,tsz);
  if(dsz>0)memcpy(g_data,fb+doff,dsz);

  // ---- find main ----
  long main_off=-1;
  for(long i=0;i<ssz/24;i++){
    long p=soff+i*24;
    if(!strcmp(fb+stroff+r32(fb,p),"main")){main_off=r64(fb,p+8);break;}
  }
  if(main_off<0){fprintf(stderr,"'main' not found\n");return 1;}
  if(g_debug)fprintf(stderr,"ELF: text=%ld longs, data=%ld bytes, main@%ld\n",
                     g_tlen,dsz,main_off);

  // ---- JIT memory ----
  long jmem_sz=64*1024*1024;
  unsigned char *jitmem=(unsigned char*)mmap(NULL,jmem_sz,
    PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANON,-1,0);
  if(jitmem==MAP_FAILED){perror("mmap");return 1;}
  jitmap=(unsigned char**)calloc(g_tlen+4,sizeof(unsigned char*));
  je=jitmem;

  // ---- entry wrapper ----
  // Callable as: long entry(long *vm_sp, char *d_base, long *unused)
  //              rdi=vm_sp  rsi=d_base  rdx=unused
  unsigned char *entry=je;
  e1(0x55);e3(0x48,0x89,0xE5); // push %rbp; mov %rsp,%rbp
  e2(0x41,0x54);                // push %r12
  e2(0x41,0x55);                // push %r13
  e2(0x41,0x56);                // push %r14
  e2(0x41,0x57);                // push %r15
  e1(0x53);                     // push %rbx  (total 6*8=48 bytes pushed -> aligned)
  e3(0x49,0x89,0xFC);           // mov %rdi,%r12  (vm_sp)
  e3(0x49,0x89,0xF5);           // mov %rsi,%r13  (d_base)
  e3(0x49,0x89,0xD6);           // mov %rdx,%r14
  e3(0x4D,0x89,0xE7);           // mov %r12,%r15  (initial bp=sp)
  e3(0x48,0x31,0xC0);           // xor %rax,%rax
  unsigned char *entry_jmp=je;
  e1(0xE9);ei32(0);             // jmp <main> -- patched below

  // ---- Pass 1: compile from pc=1 (skip padding at index 0) ----
  // Set index 0 to point at a ud2 (should never be jumped to)
  jitmap[0]=je; e2(0x0F,0x0B); // ud2

  for(long pc=1;pc<g_tlen;){
    pc=compile_one(pc)+1;
  }
  // fallthrough sentinel
  e3(0x48,0x31,0xC0); epilogue();

  // ---- Pass 2: patch branch targets ----
  for(long pc=1;pc<g_tlen;){
    long op=g_text[pc];
    if(op<0||op>EXIT){pc++;continue;}
    unsigned char *nat=jitmap[pc];

    if(op==JMP){
      long tgt=g_text[pc+1];
      unsigned char *p=nat+1;
      int rel=(int)(jitmap[tgt]-(p+4));
      p[0]=rel;p[1]=rel>>8;p[2]=rel>>16;p[3]=rel>>24;
      pc+=2;
    }
    else if(op==JSR){
      long tgt=g_text[pc+1];
      // JSR layout: mrax(10) + vpush(8) + jmp_e9(1) + imm32(4) = 23 bytes; patch at nat+19
      unsigned char *p=nat+19;
      int rel=(int)(jitmap[tgt]-(p+4));
      p[0]=rel;p[1]=rel>>8;p[2]=rel>>16;p[3]=rel>>24;
      pc+=2;
    }
    else if(op==BZ||op==BNZ){
      long tgt=g_text[pc+1];
      // layout: test(3) + 0F 84/85(2) + imm32(4); patch at nat+5
      unsigned char *p=nat+5;
      int rel=(int)(jitmap[tgt]-(p+4));
      p[0]=rel;p[1]=rel>>8;p[2]=rel>>16;p[3]=rel>>24;
      pc+=2;
    }
    else if((op>=LLA&&op<=ADJ)||op==IMMF){pc+=2;}
    else{pc++;}
  }

  // patch entry jmp to main
  {
    unsigned char *p=entry_jmp+1;
    int rel=(int)(jitmap[main_off]-(p+4));
    p[0]=rel;p[1]=rel>>8;p[2]=rel>>16;p[3]=rel>>24;
  }

  if(g_debug)fprintf(stderr,"JIT: %ld native bytes\n",(long)(je-jitmem));

  // ---- setup VM stack and run ----
  long vm_stk_sz=1024*1024;
  long *vm_stk=(long*)malloc(vm_stk_sz);
  long *vsp=(long*)((char*)vm_stk+vm_stk_sz);

  // push initial frame: argc, argv, then sentinel return address
  // exit_idx = index of the PSH+EXIT epilogue at end of linked text
  long exit_idx=g_tlen-2;
  long vm_argc=(long)(argc-ao);
  long vm_argv=(long)(argv+ao);

  *--vsp=vm_argc;
  *--vsp=vm_argv;
  *--vsp=exit_idx;  // sentinel: when main's LEV fires, it jumps to EXIT

  typedef long(*jfn_t)(long*,char*,long*);
  jfn_t jfn=(jfn_t)(void*)entry;
  return (int)jfn(vsp,g_data,g_text);
}