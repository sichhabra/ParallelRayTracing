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

    Color cuda_antiAlias(double **red,double **green,double **blue,int i,int j,int antiAliasFactor){

        double *addr_r,*addr_g,*addr_b;
        double **r,**g,**b;
        cudaMallocManaged(&addr_r,0);
        cudaMallocManaged(&addr_g,0);
        cudaMallocManaged(&addr_b,0);

        _anti_alias<<<antiAliasFactor,antiAliasFactor>>>(addr_r,addr_g,addr_b,red,green,blue,i,j,antiAliasFactor);
        
        return Color(*addr_r,*addr_g,*addr_b);
    }
}
