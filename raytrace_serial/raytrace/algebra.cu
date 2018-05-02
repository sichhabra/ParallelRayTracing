#include<stdio.h>
#include<math.h>
#include "algebra.h"
namespace Algebra{

//const double TOLERANCE = 1.0e-8;

__global__ void filter_real(int N,const std::complex<double> *inArray, double *outArray){

    int i = blockIdx.x;
    if (fabs(inArray[i].imag()) < TOLERANCE)
    {
        outArray[i] = inArray[i].real();
    }

}

int cuda_FilterRealNumbers(int numComplexValues, const std::complex<double> inArray[], double outArray[]){

    std::complex<double> *a;
    double *b;
    int N=numComplexValues;
    
    double temp[N];
    for(int i=0;i<N;i++) temp[i]=TOLERANCE;

    cudaMalloc(&a,N*sizeof(std::complex<double>));
    cudaMalloc(&b,N*sizeof(double));

    cudaMemcpy(a, inArray, (N)*sizeof(std::complex<double>), cudaMemcpyHostToDevice);
    filter_real<<<N,1>>>(N,a,b);
    
    cudaMemcpy(temp, b, (N)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(a);
    cudaFree(b);

    int count=0;
    for(int i=0;i<N;i++){
        if(temp[i]!=TOLERANCE){
            outArray[count++]=temp[i];
        }
    }

    return count;
}
}
