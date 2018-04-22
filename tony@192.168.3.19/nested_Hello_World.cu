#include <cuda_runtime.h>
__global__ void nesthelloworld(int iSize,int iDepth)
{
    unsigned int tid=threadIdx.x;
    printf("depth : %d ;blcokIdx.x : %d ;threadIdx.x :% d ",iDepth,blockIdx.x,threadIdx.x);
    if (iSize=1)
        return;
    int nthread=iSize>>1;
    if (tid=0 && nthread>0)
    {
        nesthelloworld<<<1,nthread>>>(nthread,iDepth++);
        printf("-----------> nested execution depth: %d",iDepth);
    }

}

int main(int argc,char* argv[])
{
    int grid=1;
    if (argc>=2)
        grid=atoi(argv[1]);

    nesthelloworld<<<grid,32>>>(nthread,0);
    __syncthreads();
    return 0;
}
