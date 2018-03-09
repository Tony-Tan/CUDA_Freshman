#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
  float * a=MatA;
  float * b=MatB;
  float * c=MatC;
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      c[i]=a[i]+b[i];
    }
    c+=nx;
    b+=nx;
    a+=nx;
  }
}
__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*ny;
    if (ix<nx && iy<ny)
    {
      MatC[idx]=MatA[idx]+MatB[idx];
    }
}

int main(int argc,char** argv)
{
  printf("strating...\n");
  initDevice(0);
  int nx=1<<12;
  int ny=1<<12;
  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);

  //Malloc
  float* A_host=(float*)malloc(nBytes);
  float* B_host=(float*)malloc(nBytes);
  float* C_host=(float*)malloc(nBytes);
  float* C_from_gpu=(float*)malloc(nBytes);
  initialData(A_host,nxy);
  initialData(B_host,nxy);

  //cudaMalloc
  float *A_dev=NULL;
  float *B_dev=NULL;
  float *C_dev=NULL;
  CHECK(cudaMalloc((void**)&A_dev,nBytes));
  CHECK(cudaMalloc((void**)&B_dev,nBytes));
  CHECK(cudaMalloc((void**)&C_dev,nBytes));


  CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice));

  int dimx=32;
  int dimy=32;

  // cpu compute
  cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost);
  double iStart=cpuSecond();
  sumMatrix2D_CPU(A_host,B_host,C_host,nx,ny);
  double iElaps=cpuSecond()-iStart;
  printf("CPU Execution Time elapsed %f sec\n",iElaps);

  // 2d block and 2d grid
  dim3 block_0(dimx,dimy);
  dim3 grid_0((nx-1)/block_0.x+1,(ny-1)/block_0.y+1);
  iStart=cpuSecond();
  sumMatrix<<<grid_0,block_0>>>(A_dev,B_dev,C_dev,nx,ny);
  CHECK(cudaDeviceSynchronize());
  iElaps=cpuSecond()-iStart;
  printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
        grid_0.x,grid_0.y,block_0.x,block_0.y,iElaps);
  CHECK(cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost));
  checkResult(C_host,C_from_gpu,nxy);
  // 1d block and 1d grid
  dimx=32;
  dim3 block_1(dimx);
  dim3 grid_1((nxy-1)/block_1.x+1);
  iStart=cpuSecond();
  sumMatrix<<<grid_1,block_1>>>(A_dev,B_dev,C_dev,nx*ny ,1);
  CHECK(cudaDeviceSynchronize());
  iElaps=cpuSecond()-iStart;
  printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
        grid_1.x,grid_1.y,block_1.x,block_1.y,iElaps);
  CHECK(cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost));
  checkResult(C_host,C_from_gpu,nxy);
  // 2d block and 1d grid
  dimx=32;
  dim3 block_2(dimx);
  dim3 grid_2((nx-1)/block_2.x+1,ny);
  iStart=cpuSecond();
  sumMatrix<<<grid_2,block_2>>>(A_dev,B_dev,C_dev,nx,ny);
  CHECK(cudaDeviceSynchronize());
  iElaps=cpuSecond()-iStart;
  printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
        grid_2.x,grid_2.y,block_2.x,block_2.y,iElaps);
  CHECK(cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost));
  checkResult(C_host,C_from_gpu,nxy);


  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
  free(A_host);
  free(B_host);
  free(C_host);
  free(C_from_gpu);
  cudaDeviceReset();
  return 0;
}
