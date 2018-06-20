#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define BDIMX 8
#define BDIMY 8
#define IPAD 2
//cpu transform
void transformMatrix2D_CPU(float * in,float * out,int nx,int ny)
{
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      out[i*nx+j]=in[j*nx+i];
    }
  }
}
__global__ void warmup(float * in,float * out,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*nx;
    if (ix<nx && iy<ny)
    {
      out[idx]=in[idx];
    }
}
__global__ void copyRow(float * in,float * out,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*nx;
    if (ix<nx && iy<ny)
    {
      out[idx]=in[idx];
    }
}

__global__ void transformNaiveRow(float * in,float * out,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      out[idx_col]=in[idx_row];
    }
}


//----------------------shared memory---------------------------
__global__ void transformSmem(float * in,float* out,int nx,int ny)
{
	__shared__ float tile[BDIMY][BDIMX];
	unsigned int ix,iy,transform_in_idx,transform_out_idx;
	ix=threadIdx.x+blockDim.x*blockIdx.x;
    iy=threadIdx.y+blockDim.y*blockIdx.y;
	transform_in_idx=iy*nx+ix;

	unsigned int bidx,irow,icol;
	bidx=threadIdx.y*blockDim.x+threadIdx.x;
	irow=bidx/blockDim.y;
	icol=bidx%blockDim.y;


	ix=blockIdx.y*blockDim.y+icol;
	iy=blockIdx.x*blockDim.x+irow;


	transform_out_idx=iy*ny+ix;

	if(ix<nx&& iy<ny)
	{
		tile[threadIdx.y][threadIdx.x]=in[transform_in_idx];
		__syncthreads();
		out[transform_out_idx]=tile[icol][irow];

	}

}
__global__ void transformSmemPad(float * in,float* out,int nx,int ny)
{
	__shared__ float tile[BDIMY][BDIMX+IPAD];
	unsigned int ix,iy,transform_in_idx,transform_out_idx;
	ix=threadIdx.x+blockDim.x*blockIdx.x;
    iy=threadIdx.y+blockDim.y*blockIdx.y;
	transform_in_idx=iy*nx+ix;

	unsigned int bidx,irow,icol;
	bidx=threadIdx.y*blockDim.x+threadIdx.x;
	irow=bidx/blockDim.y;
	icol=bidx%blockDim.y;


	ix=blockIdx.y*blockDim.y+icol;
	iy=blockIdx.x*blockDim.x+irow;


	transform_out_idx=iy*ny+ix;

	if(ix<nx&& iy<ny)
	{
		tile[threadIdx.y][threadIdx.x]=in[transform_in_idx];
		__syncthreads();
		out[transform_out_idx]=tile[icol][irow];

	}

}
__global__ void transformSmemUnrollPad(float * in,float* out,int nx,int ny)
{
	__shared__ float tile[BDIMY*(BDIMX*2+IPAD)];


	unsigned int ix,iy,transform_in_idx,transform_out_idx;
	ix=threadIdx.x+blockDim.x*blockIdx.x*2;
    iy=threadIdx.y+blockDim.y*blockIdx.y;
	transform_in_idx=iy*nx+ix;

	unsigned int bidx,irow,icol;
	bidx=threadIdx.y*blockDim.x+threadIdx.x;
	irow=bidx/blockDim.y;
	icol=bidx%blockDim.y;


	unsigned int ix2=blockIdx.y*blockDim.y+icol;
	unsigned int iy2=blockIdx.x*blockDim.x*2+irow;


	transform_out_idx=iy2*ny+ix2;

	if(ix+blockDim.x<nx&& iy<ny)
	{
		unsigned int row_idx=threadIdx.y*(blockDim.x*2+IPAD)+threadIdx.x;
		tile[row_idx]=in[transform_in_idx];
		tile[row_idx+BDIMX]=in[transform_in_idx+BDIMX];
		__syncthreads();
		unsigned int col_idx=icol*(blockDim.x*2+IPAD)+irow;
        out[transform_out_idx]=tile[col_idx];
		out[transform_out_idx+ny*BDIMX]=tile[col_idx+BDIMX];

	}

}

//--------------------------------------------------------------



int main(int argc,char** argv)
{
  printf("strating...\n");
  initDevice(0);
  int nx=1<<12;
  int ny=1<<12;
  int dimx=BDIMX;
  int dimy=BDIMY;
  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);
  int transform_kernel=0;
  if(argc==2)
    transform_kernel=atoi(argv[1]);
  if(argc>=4)
  {
      transform_kernel=atoi(argv[1]);
      dimx=atoi(argv[2]);
      dimy=atoi(argv[3]);
  }

  //Malloc
  float* A_host=(float*)malloc(nBytes);
  float* B_host_cpu=(float*)malloc(nBytes);
  float* B_host=(float*)malloc(nBytes);
  initialData(A_host,nxy);

  //cudaMalloc
  float *A_dev=NULL;
  float *B_dev=NULL;
  CHECK(cudaMalloc((void**)&A_dev,nBytes));
  CHECK(cudaMalloc((void**)&B_dev,nBytes));

  CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(B_dev,0,nBytes));



  // cpu compute
  double iStart=cpuSecond();
  transformMatrix2D_CPU(A_host,B_host_cpu,nx,ny);
  double iElaps=cpuSecond()-iStart;
  printf("CPU Execution Time elapsed %f sec\n",iElaps);

  // 2d block and 2d grid
  dim3 block(dimx,dimy);
  dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
  dim3 block_1(dimx,dimy);
  dim3 grid_1((nx-1)/(block_1.x*2)+1,(ny-1)/block_1.y+1);
  //warmup
  warmup<<<grid,block>>>(A_dev,B_dev,nx,ny);
  CHECK(cudaDeviceSynchronize());
  iStart=cpuSecond();
  switch(transform_kernel)
  {
  case 0:
	    copyRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
        printf("copyRow ");
	    break;
  case 1:
	    transformNaiveRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
        printf("transformNaiveRow ");
	    break;
  case 2:
  		transformSmem<<<grid,block>>>(A_dev,B_dev,nx,ny);
        printf("transformSmem ");
		break;
  case 3:
		transformSmemPad<<<grid,block>>>(A_dev,B_dev,nx,ny);
        printf("transformSmemPad ");
		break;
  case 4:
        transformSmemUnrollPad<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
        printf("transformSmemUnrollPad ");
        break;
  default:
    break;
  }
  CHECK(cudaDeviceSynchronize());
  iElaps=cpuSecond()-iStart;
  printf(" Time elapsed %f sec\n",iElaps);
  CHECK(cudaMemcpy(B_host,B_dev,nBytes,cudaMemcpyDeviceToHost));
  checkResult(B_host,B_host_cpu,nxy);

  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A_host);
  free(B_host);
  free(B_host_cpu);
  cudaDeviceReset();
  return 0;
}
