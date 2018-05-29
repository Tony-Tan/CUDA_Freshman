#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1
__global__ void setRowReadRow(float * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}
__global__ void setColReadCol(float * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.x][threadIdx.y]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setColReadRow(float * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.x][threadIdx.y]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}
__global__ void setRowReadCol(float * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDyn(float * out)
{
    extern __shared__ int tile[];
    unsigned int row_idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int col_idx=threadIdx.x*blockDim.y+threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}
__global__ void setRowReadColIpad(float * out)
{
    __shared__ int tile[BDIMY][BDIMX+IPAD];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDynIpad(float * out)
{
    extern __shared__ int tile[];
    unsigned int row_idx=threadIdx.y*(blockDim.x+1)+threadIdx.x;
    unsigned int col_idx=threadIdx.x*(blockDim.x+1)+threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}
//--------------------rectagle---------------------
__global__ void setRowReadColRect(float * out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[icol][irow];
}
__global__ void setRowReadColRectDyn(float * out)
{
    extern __shared__ int tile[];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    unsigned int col_idx=icol*blockDim.x+irow;
    tile[idx]=idx;
    __syncthreads();
    out[idx]=tile[col_idx];
}
__global__ void setRowReadColRectPad(float * out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT+IPAD];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[icol][irow];
}
__global__ void setRowReadColRectDynPad(float * out)
{
    extern __shared__ int tile[];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    unsigned int row_idx=threadIdx.y*(IPAD+blockDim.x)+threadIdx.x;
    unsigned int col_idx=icol*(IPAD+blockDim.x)+irow;
    tile[row_idx]=idx;
    __syncthreads();
    out[idx]=tile[col_idx];
}
int main(int argc,char **argv)
{
  // set up device
  initDevice(0);
  int kernel=0;
  if(argc>=2)
    kernel=atoi(argv[1]);
  int nElem=BDIMX*BDIMY;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *res_from_device=(float*)malloc(nByte);
  memset(res_from_device,0,nByte);
  float * out;
  CHECK(cudaMalloc((float**)&out,nByte));


  dim3 block(BDIMY,BDIMX);
  dim3 grid(1,1);
  dim3 block_rect(BDIMY_RECT,BDIMX_RECT);
  dim3 grid_rect(1,1);
  double iStart,iElaps;
  iStart=cpuSecond();
  switch(kernel)
  {
      case 0:
      setRowReadRow<<<grid,block>>>(out);
      break;
      case 1:
      setColReadCol<<<grid,block>>>(out);
      break;
      case 2:
      setColReadRow<<<grid,block>>>(out);
      break;
      case 3:
      setRowReadCol<<<grid,block>>>(out);
      break;
      case 4:
      setRowReadColDyn<<<grid,block,(BDIMX)*BDIMY*sizeof(float)>>>(out);
      break;
      case 5:
      setRowReadColIpad<<<grid,block>>>(out);
      break;
      case 6:
      setRowReadColDynIpad<<<grid,block,(BDIMX+IPAD)*BDIMY*sizeof(float)>>>(out);
      break;
      case 7:
      setRowReadColRect<<<grid_rect,block_rect>>>(out);
      break;
      case 8:
      setRowReadColRectDyn<<<grid_rect,block_rect,(BDIMX)*BDIMY*sizeof(float)>>>(out);
      break;
      case 9:
      setRowReadColRectPad<<<grid_rect,block_rect>>>(out);
      break;
      case 10:
      setRowReadColRectDynPad<<<grid_rect,block_rect,(BDIMX+1)*BDIMY*sizeof(float)>>>(out);
      break;

  }
  iElaps=cpuSecond()-iStart;
  printf("Execution Time elapsed %f sec\n",iElaps);
  CHECK(cudaMemcpy(res_from_device,out,nByte,cudaMemcpyDeviceToHost));
  cudaFree(out);
  free(res_from_device);
  return 0;
}
