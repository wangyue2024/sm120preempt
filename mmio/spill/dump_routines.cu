#include "arena.h"
#include <cuda_runtime.h>
__global__ void k(){}
int main(){
  cudaFree(0); k<<<1,1>>>(); cudaDeviceSynchronize();
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  auto I = sm120arena::probe(ctx,0);
  // routines relative to handler_va: 0x1200 reason,0x2400 bit0,0x2f00 bit1,0x3900 bit2,0x4400 bit3(tools),0x5700 bit4,0x6200 bit5,0x6c00 bit6,0x7700 bit7
  uint64_t offs[] = {0x1200,0x2400,0x2f00,0x3900,0x4400,0x5700,0x6200,0x6c00,0x7700};
  for (auto o: offs){
    uint64_t va = I.handler_va + o;
    cuxtra::etbl::CUtoolsMemObjHandle_st *ob=nullptr; unsigned long long off=0; unsigned long sz=0;
    bool m = cuxtra::cuda::FindMemObjHandle(ctx, va, &ob, &off, &sz);
    printf("routine +%#llx va=%#llx memobj=%d off=%#llx size=%#lx\n",(unsigned long long)o,(unsigned long long)va,(int)m,off,sz);
    if(!m) continue;
    size_t n = sz - off; if (n > 0x1400) n = 0x1400;
    std::vector<uint8_t> b(n);
    size_t got = cuXtraMemcpyDtoH(b.data(), va, n, 0);
    char fn[64]; snprintf(fn,64,"r_%05llx.bin",(unsigned long long)o);
    FILE*f=fopen(fn,"wb"); fwrite(b.data(),1,got,f); fclose(f);
  }
  return 0;
}
