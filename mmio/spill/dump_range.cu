#include "arena.h"
#include <cuda_runtime.h>
__global__ void k(){}
int main(int argc,char**argv){
  uint64_t lo=strtoull(argv[1],0,0), hi=strtoull(argv[2],0,0);
  cudaFree(0); k<<<1,1>>>(); cudaDeviceSynchronize();
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  auto I = sm120arena::probe(ctx,0);
  printf("handler_va=%#llx base=%#llx\n",(unsigned long long)I.handler_va,(unsigned long long)I.base);
  uint64_t o=lo;
  while(o<hi){
    uint64_t va=I.handler_va+o;
    cuxtra::etbl::CUtoolsMemObjHandle_st *ob=nullptr; unsigned long long off=0; unsigned long sz=0;
    bool m=cuxtra::cuda::FindMemObjHandle(ctx,va,&ob,&off,&sz);
    if(!m){ o+=0x100; continue;}
    printf("obj at +%#llx objbase=+%#llx size=%#lx\n",(unsigned long long)o,(unsigned long long)(off-I.handler_va),sz); va=off; o=off-I.handler_va;
    std::vector<uint8_t> b(sz); size_t got=cuXtraMemcpyDtoH(b.data(),va,sz,0);
    char fn[64]; snprintf(fn,64,"r_%05llx.bin",(unsigned long long)o);
    FILE*f=fopen(fn,"wb"); fwrite(b.data(),1,got,f); fclose(f);
    o+= (sz+0xff)&~0xffULL;
  }
}
