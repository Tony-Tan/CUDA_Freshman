#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define BDIM 16
#define SEGM 4
__global__ void test_shfl_broadcast(int *in,int*out,int const srcLans)
{
    int value=in[threadIdx.x];
    value=__shfl(value,srcLans,BDIM);
    out[threadIdx.x]=value;

}

__global__ void test_shfl_up(int *in,int*out,int const delta)
{
    int value=in[threadIdx.x];
    value=__shfl_up(value,delta,BDIM);
    out[threadIdx.x]=value;

}

__global__ void test_shfl_down(int *in,int*out,int const delta)
{
    int value=in[threadIdx.x];
    value=__shfl_down(value,delta,BDIM);
    out[threadIdx.x]=value;

}

__global__ void test_shfl_wrap(int *in,int*out,int const offset)
{
    int value=in[threadIdx.x];
    value=__shfl(value,threadIdx.x+offset,BDIM);
    out[threadIdx.x]=value;

}

__global__ void test_shfl_xor(int *in,int*out,int const mask)
{
    int value=in[threadIdx.x];
    value=__shfl_xor(value,mask,BDIM);
    out[threadIdx.x]=value;

}

__global__ void test_shfl_xor_array(int *in,int*out,int const mask)
{
    int idx=threadIdx.x*SEGM;
    int value[SEGM];
    for(int i=0;i<SEGM;i++)
        value[i]=in[idx+i];
    value[0]=__shfl_xor(value[0],mask,BDIM);
    value[1]=__shfl_xor(value[1],mask,BDIM);
    value[2]=__shfl_xor(value[2],mask,BDIM);
    value[3]=__shfl_xor(value[3],mask,BDIM);
    for(int i=0;i<SEGM;i++)
        out[idx+i]=value[i];

}
__inline__ __device__
void swap(int *value,int laneIdx,int mask,int firstIdx,int secondIdx)
{
    bool pred=((laneIdx%(2))==0);
    if(pred)
    {
        int tmp=value[firstIdx];
        value[firstIdx]=value[secondIdx];
        value[secondIdx]=tmp;

    }
    value[secondIdx]=__shfl_xor(value[secondIdx],mask,BDIM);
    if(pred)
    {
        int tmp=value[firstIdx];
        value[firstIdx]=value[secondIdx];
        value[secondIdx]=tmp;
    }
}

__global__ void test_shfl_swap(int *in,int* out,int const mask,int firstIdx,int secondIdx)
{
    int idx=threadIdx.x*SEGM;
    int value[SEGM];
    for(int i=0;i<SEGM;i++)
        value[i]=in[idx+i];
    swap(value,threadIdx.x,mask,firstIdx,secondIdx);
    for(int i=0;i<SEGM;i++)
        out[idx+i]=value[i];

}

int main(int argc,char** argv)
{
    printf("strating...\n");
    initDevice(0);
    int dimx=BDIM;
    unsigned int data_size=BDIM;
    int nBytes=data_size*sizeof(int);
    int kernel_num=0;
    if(argc>=2)
        kernel_num=atoi(argv[1]);

    //Malloc
    //int * in_host=(int*)malloc(nBytes);
    int in_host[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int * out_gpu=(int*)malloc(nBytes);
    //initialData_int(in_host,data_size);

    //cudaMalloc
    int * in_dev=NULL;
    int * out_dev=NULL;

    CHECK(cudaMalloc((void**)&in_dev,nBytes));
    CHECK(cudaMalloc((void**)&out_dev,nBytes));
    CHECK(cudaMemcpy(in_dev,in_host,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out_dev,0,nBytes));


    // test  _shfl broadcast
    dim3 block(dimx);
    dim3 grid((data_size-1)/block.x+1);
    switch(kernel_num)
    {
        case 0:
            test_shfl_broadcast<<<grid,block>>>(in_dev,out_dev,2);
            printf("test_shfl_broadcast\n");
            break;
        case 1:
            test_shfl_up<<<grid,block>>>(in_dev,out_dev,2);
            printf("test_shfl_up\n");
            break;
        case 2:
            test_shfl_down<<<grid,block>>>(in_dev,out_dev,2);
            printf("test_shfl_down\n");
            break;
        case 3:
            test_shfl_wrap<<<grid,block>>>(in_dev,out_dev,2);
            printf("test_shfl_wrap\n");
            break;
        case 4:
            test_shfl_xor<<<grid,block>>>(in_dev,out_dev,1);
            printf("test_shfl_xor\n");
            break;
        case 5:
            test_shfl_xor_array<<<1,block.x/SEGM>>>(in_dev,out_dev,1);
            printf("test_shfl_xor_array\n");
            break;
        case 6:
            test_shfl_swap<<<1,block.x/SEGM>>>(in_dev,out_dev,1,0,3);
            printf("test_shfl_swap\n");
            break;
        default:
            break;
    }
    CHECK(cudaMemcpy(out_gpu,out_dev,nBytes,cudaMemcpyDeviceToHost));
    //show result
    printf("input:\t");
    for(int i=0;i<data_size;i++)
        printf("%4d ",in_host[i]);
    printf("\n\n\n\n\noutput:\t");
    for(int i=0;i<data_size;i++)
        printf("%4d ",out_gpu[i]);
    printf("\n");
    CHECK(cudaMemset(out_dev,0,nBytes));
    // stencil 1d read only


    cudaFree(in_dev);
    cudaFree(out_dev);
    free(out_gpu);
    //free(in_host);
    cudaDeviceReset();
    return 0;
}
