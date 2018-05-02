#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"


void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}
__global__ void sumArraysGPU(float*a,float*b,float*res)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);

  int nElem=1<<14;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *res_from_gpu_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_host,*b_host,*res_d;
  CHECK(cudaHostAlloc((float**)&a_host,nByte,cudaHostAllocMapped));
  CHECK(cudaHostAlloc((float**)&b_host,nByte,cudaHostAllocMapped));
  CHECK(cudaMalloc((float**)&res_d,nByte));
  res_from_gpu_h=(float*)malloc(nByte);

  initialData(a_host,nElem);
  initialData(b_host,nElem);

  dim3 block(1024);
  dim3 grid(nElem/block.x);
  sumArraysGPU<<<grid,block>>>(a_host,b_host,res_d);
  printf("Execution configuration<<<%d,%d>>>\n",grid.x,block.x);

  CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  sumArrays(a_host,b_host,res_h,nElem);

  checkResult(res_h,res_from_gpu_h,nElem);
  cudaFreeHost(a_host);
  cudaFreeHost(b_host);
  cudaFree(res_d);

  free(res_h);
  free(res_from_gpu_h);

  return 0;
}
