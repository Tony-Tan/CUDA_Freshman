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
__global__ void sumArraysGPU(float*a,float*b,float*res,int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i < N)
    res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  // set up device
  initDevice(0);

  int nElem=1<<24;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *res_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  //memset(res_from_gpu_h,0,nByte);

  float *a_d,*b_d,*res_d;
  CHECK(cudaMallocManaged((float**)&a_d,nByte));
  CHECK(cudaMallocManaged((float**)&b_d,nByte));
  CHECK(cudaMallocManaged((float**)&res_d,nByte));

  initialData(a_d,nElem);
  initialData(b_d,nElem);

  //CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  //CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  dim3 block(512);
  dim3 grid((nElem-1)/block.x+1);

  double iStart,iElaps;
  iStart=cpuSecond();
  sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d,nElem);
  cudaDeviceSynchronize();
  iElaps=cpuSecond()-iStart;
  printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

  //CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  sumArrays(b_d,b_d,res_h,nElem);

  checkResult(res_h,res_d,nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(res_h);

  return 0;
}
