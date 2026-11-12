// dump the live sm_120 .text.traphandler and slot-20 tools image to files
#include "arena.h"
#include <cuda_runtime.h>
__global__ void k(){}
int main(){
  cudaFree(0); k<<<1,1>>>(); cudaDeviceSynchronize();
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  auto I = sm120arena::probe(ctx,0);
  sm120arena::report(I);
  FILE*f=fopen("live_handler.bin","wb"); fwrite(I.handler.data(),1,I.handler.size(),f); fclose(f);
  f=fopen("tools_image.bin","wb"); fwrite(I.tools.data(),1,I.tools.size(),f); fclose(f);
  printf("handler_va=%#llx tools_pc=%#llx\n",(unsigned long long)I.handler_va,(unsigned long long)I.tools_pc);
  return 0;
}
