#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define TEMPLATE_SIZE 9
#define TEMP_RADIO_SIZE (TEMPLATE_SIZE/2)
#define BDIM 32

__constant__ float coef[TEMP_RADIO_SIZE];//if in midle of the program will be error
void convolution(float *in,float *out,float* template_,const unsigned int array_size)
{
    for(int i=TEMP_RADIO_SIZE;i<array_size-TEMP_RADIO_SIZE;i++)
    {
        for(int j=1;j<=TEMP_RADIO_SIZE;j++)
        {
            out[i]+=template_[j-1]*(in[i+j]-in[i-j]);
        }

        //printf("%d:CPU :%lf\n",i,out[i]);
    }

}

__global__ void stencil_1d(float * in,float * out)
{
    __shared__ float smem[BDIM+2*TEMP_RADIO_SIZE];
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    int sidx=threadIdx.x+TEMP_RADIO_SIZE;
    smem[sidx]=in[idx];

    if (threadIdx.x<TEMP_RADIO_SIZE)

    {
        if(idx>TEMP_RADIO_SIZE)
            smem[sidx-TEMP_RADIO_SIZE]=in[idx-TEMP_RADIO_SIZE];
        if(idx<gridDim.x*blockDim.x-BDIM)
            smem[sidx+BDIM]=in[idx+BDIM];

    }

    __syncthreads();
    if (idx<TEMP_RADIO_SIZE||idx>=gridDim.x*blockDim.x-TEMP_RADIO_SIZE)
        return;
    float temp=.0f;
    #pragma unroll
    for(int i=1;i<=TEMP_RADIO_SIZE;i++)
    {
        temp+=coef[i-1]*(smem[sidx+i]-smem[sidx-i]);
    }
    out[idx]=temp;
    //printf("%d:GPU :%lf,\n",idx,temp);
}
//read only
__global__ void stencil_1d_readonly(float * in,float * out,const float* __restrict__ dcoef)
{
    __shared__ float smem[BDIM+2*TEMP_RADIO_SIZE];
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    int sidx=threadIdx.x+TEMP_RADIO_SIZE;
    smem[sidx]=in[idx];

    if (threadIdx.x<TEMP_RADIO_SIZE)

    {
        if(idx>TEMP_RADIO_SIZE)
            smem[sidx-TEMP_RADIO_SIZE]=in[idx-TEMP_RADIO_SIZE];
        if(idx<gridDim.x*blockDim.x-BDIM)
            smem[sidx+BDIM]=in[idx+BDIM];

    }

    __syncthreads();
    if (idx<TEMP_RADIO_SIZE||idx>=gridDim.x*blockDim.x-TEMP_RADIO_SIZE)
        return;
    float temp=.0f;
    #pragma unroll
    for(int i=1;i<=TEMP_RADIO_SIZE;i++)
    {
        temp+=dcoef[i-1]*(smem[sidx+i]-smem[sidx-i]);
    }
    out[idx]=temp;
    //printf("%d:GPU :%lf,\n",idx,temp);
}

int main(int argc,char** argv)
{
    printf("strating...\n");
    initDevice(0);
    int dimx=BDIM;
    unsigned int nxy=1<<16;
    int nBytes=nxy*sizeof(float);


    //Malloc
    float* in_host=(float*)malloc(nBytes);
    float* out_gpu=(float*)malloc(nBytes);
    float* out_cpu=(float*)malloc(nBytes);
    memset(out_cpu,0,nBytes);
    initialData(in_host,nxy);

    //cudaMalloc
    float *in_dev=NULL;
    float *out_dev=NULL;

    initialData(in_host,nxy);
    float templ_[]={-1.0,-2.0,2.0,1.0};
    CHECK(cudaMemcpyToSymbol(coef,templ_,TEMP_RADIO_SIZE*sizeof(float)));

    CHECK(cudaMalloc((void**)&in_dev,nBytes));
    CHECK(cudaMalloc((void**)&out_dev,nBytes));
    CHECK(cudaMemcpy(in_dev,in_host,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out_dev,0,nBytes));



    // cpu compute
    double iStart=cpuSecond();
    convolution(in_host,out_cpu,templ_,nxy);
    double iElaps=cpuSecond()-iStart;
    //printf("CPU Execution Time elapsed %f sec\n",iElaps);

    // stencil 1d
    dim3 block(dimx);
    dim3 grid((nxy-1)/block.x+1);
    stencil_1d<<<grid,block>>>(in_dev,out_dev);
    CHECK(cudaDeviceSynchronize());
    iElaps=cpuSecond()-iStart;
    printf("stencil_1d Time elapsed %f sec\n",iElaps);
    CHECK(cudaMemcpy(out_gpu,out_dev,nBytes,cudaMemcpyDeviceToHost));
    checkResult(out_cpu,out_gpu,nxy);
    CHECK(cudaMemset(out_dev,0,nBytes));
    // stencil 1d read only
    float * dcoef_ro;
    CHECK(cudaMalloc((void**)&dcoef_ro,TEMP_RADIO_SIZE * sizeof(float)));
    CHECK(cudaMemcpy(dcoef_ro,templ_,TEMP_RADIO_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    stencil_1d_readonly<<<grid,block>>>(in_dev,out_dev,dcoef_ro);
    CHECK(cudaDeviceSynchronize());
    iElaps=cpuSecond()-iStart;
    printf("stencil_1d_readonly Time elapsed %f sec\n",iElaps);
    CHECK(cudaMemcpy(out_gpu,out_dev,nBytes,cudaMemcpyDeviceToHost));
    checkResult(out_cpu,out_gpu,nxy);

    cudaFree(dcoef_ro);
    cudaFree(in_dev);
    cudaFree(out_dev);
    free(out_gpu);
    free(out_cpu);
    free(in_host);
    cudaDeviceReset();
    return 0;
}
