#include "imager.h"
#include "antialias.h"

namespace Imager{
#if __CUDA_ARCH__ < 600
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                        __longlong_as_double(assumed)));

            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
#endif 
    __global__ void _anti_alias(double *addr_r,double *addr_g,double *addr_b,double **red,double **green,double **blue,int i,int j,int antiAliasFactor){
        int di=threadIdx.x;
        int dj=threadIdx.y;
        int x=antiAliasFactor*i + di;
        int y=antiAliasFactor*j + dj;
        atomicAdd(addr_r,red[x][y]);
        atomicAdd(addr_g,green[x][y]);
        atomicAdd(addr_b,blue[x][y]);
    }

    Color cuda_antiAlias(double **red,double **green,double **blue,int i,int j,int antiAliasFactor,int wide,int height){

        double *addr_r,*addr_g,*addr_b,*rr,*gg,*bb;
        double **r,**g,**b;
        cudaMallocManaged(&addr_r,4);
        cudaMallocManaged(&addr_g,4);
        cudaMallocManaged(&addr_b,4);
        cudaMalloc(&r,wide*height*sizeof(double));
        cudaMalloc(&g,wide*height*sizeof(double));
        cudaMalloc(&b,wide*height*sizeof(double));
        cudaMemset(addr_r,0,sizeof(double));
        cudaMemset(addr_g,0,sizeof(double));
        cudaMemset(addr_b,0,sizeof(double));

        cudaMemcpy(r,red,wide*height*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(g,green,wide*height*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(b,blue,wide*height*sizeof(double),cudaMemcpyHostToDevice);

        _anti_alias<<<antiAliasFactor,antiAliasFactor>>>(addr_r,addr_g,addr_b,red,green,blue,i,j,antiAliasFactor);

	rr=(double*)malloc(sizeof(double));
	gg=(double*)malloc(sizeof(double));
	bb=(double*)malloc(sizeof(double));

	cudaMemcpy(rr,addr_r,sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(gg,addr_g,sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(bb,addr_b,sizeof(double),cudaMemcpyDeviceToHost);
        
        return Color(*rr,*gg,*bb);
    }
}
