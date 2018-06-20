#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define DIM 1024

int recursiveReduce(int *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce(data, stride);
}
__global__ void warmup(int * g_idata, int * g_odata, unsigned int n)
{

	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}


__global__ void reduceGmem(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x;

	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		idata[tid]+=idata[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		idata[tid]+=idata[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		idata[tid]+=idata[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		idata[tid]+=idata[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}


__global__ void reduceSmem(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
    __shared__ int smem[DIM];
	unsigned int tid = threadIdx.x;
	//unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x;

    smem[tid]=idata[tid];
	__syncthreads();
	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		smem[tid]+=smem[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		smem[tid]+=smem[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		smem[tid]+=smem[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		smem[tid]+=smem[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vsmem = smem;
		vsmem[tid]+=vsmem[tid+32];
		vsmem[tid]+=vsmem[tid+16];
		vsmem[tid]+=vsmem[tid+8];
		vsmem[tid]+=vsmem[tid+4];
		vsmem[tid]+=vsmem[tid+2];
		vsmem[tid]+=vsmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = smem[0];

}

__global__ void reduceUnroll4Smem(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
    __shared__ int smem[DIM];
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*4+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
    int tempSum=0;
	if(idx+3 * blockDim.x<=n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		tempSum=a1+a2+a3+a4;

	}
    smem[tid]=tempSum;
	__syncthreads();
	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		smem[tid]+=smem[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		smem[tid]+=smem[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		smem[tid]+=smem[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		smem[tid]+=smem[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vsmem = smem;
		vsmem[tid]+=vsmem[tid+32];
		vsmem[tid]+=vsmem[tid+16];
		vsmem[tid]+=vsmem[tid+8];
		vsmem[tid]+=vsmem[tid+4];
		vsmem[tid]+=vsmem[tid+2];
		vsmem[tid]+=vsmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = smem[0];

}

int main(int argc,char** argv)
{
	initDevice(0);

	bool bResult = false;
	//initialization

	int size = 1 << 24;
	printf("	with array size %d  \n", size);

	//execution configuration
	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int * tmp = (int*)malloc(bytes);

	//initialize the array
	initialData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double iStart, iElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	//cpu reduction
	int cpu_sum = 0;
	iStart = cpuSecond();
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	iElaps = cpuSecond() - iStart;
	printf("cpu reduce           elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);


	//kernel 1:warmup
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	warmup <<<grid.x/2, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("gpu warmup           elapsed %lf ms\n",iElaps);



	//reduceGmem
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceGmem <<<grid.x, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("reduceGmem           elapsed %lf ms gpu_sum: %d\n",iElaps, gpu_sum);

    //reduceSmem
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceSmem <<<grid.x, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("reduceSmem           elapsed %lf ms gpu_sum: %d\n",iElaps, gpu_sum);



    //reduceUnroll4Smem
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceUnroll4Smem <<<grid.x/4, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/4; i++)
		gpu_sum += odata_host[i];
	printf("reduceUnroll4Smem    elapsed %lf ms gpu_sum: %d\n",iElaps, gpu_sum);


	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));
	//reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;

}
