#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define N 300000
__global__ void kernel_1()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_2()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_3()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
__global__ void kernel_4()
{
    double sum=0.0;
    for(int i=0;i<N;i++)
        sum=sum+tan(0.1)*tan(0.1);
}
int main()
{
    int n_stream=4;
    cudaStream_t *stream=(cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
    for(int i=0;i<n_stream;i++)
    {
        cudaStreamCreate(&stream[i]);
    }
    dim3 block(1);
    dim3 grid(1);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<n_stream;i++)
    {
        kernel_1<<<grid,block,0,stream[i]>>>();
        kernel_2<<<grid,block,0,stream[i]>>>();
        kernel_3<<<grid,block>>>();
        kernel_4<<<grid,block,0,stream[i]>>>();
    }
    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    
    for(int i=0;i<n_stream;i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(stream);
    return 0;
}