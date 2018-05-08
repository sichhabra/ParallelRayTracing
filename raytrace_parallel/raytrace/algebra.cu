#include<stdio.h>
#include<math.h>
#include "algebra.h"
namespace Algebra{

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
    
    cudaMalloc(&a,N*sizeof(std::complex<double>));
    cudaMalloc(&b,N*sizeof(double));

    cudaMemcpy(a, inArray, (N)*sizeof(std::complex<double>), cudaMemcpyHostToDevice);
    filter_real<<<N,1>>>(N,a,b);
    
    cudaMemcpy(outArray, b, (N)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(a);
    cudaFree(b);

    return N;
}
}
