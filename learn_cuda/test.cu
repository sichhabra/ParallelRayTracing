#include<iostream>
using namespace std;

__global__ void matrixAdd(int **a,int **b,int **c){
    
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    c[i][j] = a[i][j]+b[i][j];
}

int main(){
    int N=16;
    int A[N][N],B[N][N],C[N][N];
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            A[i][j]=1;
            B[i][j]=1;
            C[i][j]=0;
        }
    }

    int **a,**b,**c;
    cudaMalloc((void**)&a,N*N*sizeof(int));
    cudaMalloc((void**)&b,N*N*sizeof(int));
    cudaMalloc((void**)&c,N*N*sizeof(int));
    
    cudaMemcpy(a, A, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blocksPerGrid(N/16,N/16,1);
    dim3 threadsPerBlock(16,16,1);
    matrixAdd<<<blocksPerGrid,threadsPerBlock>>>(a,b,c);

    cudaMemcpy(C, c, (N*N)*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<C[i][j]<<" ";
        }
        cout<<endl;
    }

}
