#ifndef FRESHMAN_H
#define FRESHMAN_H
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#include <sys/time.h>

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}
void initialData(float* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)/1000.0f;
  }
}
void printMatrix(float * C,const int nx,const int ny)
{
  float *ic=C;
  printf("Matrix<%d,%d>:",ny,nx);
  for(int i=0;i<ny;i++)
  {
    for(int j=0;j<nx;j++)
    {
      printf("%6f ",C[j]);
    }
    ic+=nx;
    printf("\n");
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}
void checkResult(float * hostRef,float * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])",hostRef[i],i,gpuRef[i],i);
      break;
    }
  }
  printf("Check result success!\n");
}
#endif//FRESHMAN_H
