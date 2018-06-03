#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1
__global__ void warmup(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}
__global__ void setRowReadRow(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}
__global__ void setColReadCol(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.x][threadIdx.y]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setColReadRow(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.x][threadIdx.y]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.y][threadIdx.x];
}
__global__ void setRowReadCol(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDyn(int * out)
{
    extern __shared__ int tile[];
    unsigned int row_idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int col_idx=threadIdx.x*blockDim.y+threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}
__global__ void setRowReadColIpad(int * out)
{
    __shared__ int tile[BDIMY][BDIMX+IPAD];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;

    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDynIpad(int * out)
{
    extern __shared__ int tile[];
    unsigned int row_idx=threadIdx.y*(blockDim.x+1)+threadIdx.x;
    unsigned int col_idx=threadIdx.x*(blockDim.x+1)+threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}
//--------------------rectagle---------------------
__global__ void setRowReadColRect(int * out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[icol][irow];
}
__global__ void setRowReadColRectDyn(int * out)
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
__global__ void setRowReadColRectPad(int * out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT+IPAD*2];
    unsigned int idx=threadIdx.y*blockDim.x+threadIdx.x;
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x]=idx;
    __syncthreads();
    out[idx]=tile[icol][irow];
}
__global__ void setRowReadColRectDynPad(int * out)
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
  int nByte=sizeof(int)*nElem;
  int * out;
  CHECK(cudaMalloc((int**)&out,nByte));
  cudaSharedMemConfig MemConfig;
  CHECK(cudaDeviceGetSharedMemConfig(&MemConfig));
  printf("--------------------------------------------\n");
  switch (MemConfig) {

      case cudaSharedMemBankSizeFourByte:
        printf("the device is cudaSharedMemBankSizeFourByte: 4-Byte\n");
      break;
      case cudaSharedMemBankSizeEightByte:
        printf("the device is cudaSharedMemBankSizeEightByte: 8-Byte\n");
      break;

  }
  printf("--------------------------------------------\n");
  dim3 block(BDIMY,BDIMX);
  dim3 grid(1,1);
  dim3 block_rect(BDIMX_RECT,BDIMY_RECT);
  dim3 grid_rect(1,1);
  warmup<<<grid,block>>>(out);
  printf("warmup!\n");
  double iStart,iElaps;
  iStart=cpuSecond();
  switch(kernel)
  {
      case 0:
          {
          setRowReadRow<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadRow  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
      //break;
      //case 1:
          iStart=cpuSecond();
          setColReadCol<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setColReadCol  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
        }
      case 2:
        {
          setColReadRow<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setColReadRow  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
        }
      case 3:
      {
          setRowReadCol<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadCol  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 4:
      {
            setRowReadColDyn<<<grid,block,(BDIMX)*BDIMY*sizeof(int)>>>(out);
            cudaDeviceSynchronize();
            iElaps=cpuSecond()-iStart;
            printf("setRowReadColDyn  ");
            printf("Execution Time elapsed %f sec\n",iElaps);
            break;
        }
      case 5:
      {
          setRowReadColIpad<<<grid,block>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColIpad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 6:
      {
          setRowReadColDynIpad<<<grid,block,(BDIMX+IPAD)*BDIMY*sizeof(int)>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColDynIpad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 7:
      {
          setRowReadColRect<<<grid_rect,block_rect>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRect  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 8:
      {
          setRowReadColRectDyn<<<grid_rect,block_rect,(BDIMX)*BDIMY*sizeof(int)>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRectDyn  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 9:
      {
          setRowReadColRectPad<<<grid_rect,block_rect>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRectPad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 10:
      {
          setRowReadColRectDynPad<<<grid_rect,block_rect,(BDIMX+1)*BDIMY*sizeof(int)>>>(out);
          cudaDeviceSynchronize();
          iElaps=cpuSecond()-iStart;
          printf("setRowReadColRectDynPad  ");
          printf("Execution Time elapsed %f sec\n",iElaps);
          break;
      }
      case 11:
      {
            setRowReadRow<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setColReadCol<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setColReadRow<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setRowReadCol<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setRowReadColDyn<<<grid,block,(BDIMX)*BDIMY*sizeof(int)>>>(out);
            cudaDeviceSynchronize();

            setRowReadColIpad<<<grid,block>>>(out);
            cudaDeviceSynchronize();

            setRowReadColDynIpad<<<grid,block,(BDIMX+IPAD)*BDIMY*sizeof(int)>>>(out);
            cudaDeviceSynchronize();
            break;
    }
    case 12:
    {
        setRowReadColRect<<<grid_rect,block_rect>>>(out);
        setRowReadColRectDyn<<<grid_rect,block_rect,(BDIMX)*BDIMY*sizeof(int)>>>(out);
        setRowReadColRectPad<<<grid_rect,block_rect>>>(out);
        setRowReadColRectDynPad<<<grid_rect,block_rect,(BDIMX+1)*BDIMY*sizeof(int)>>>(out);
        break;
    }

  }

  cudaFree(out);
  return 0;
}
