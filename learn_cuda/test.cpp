#include<iostream>
using namespace std;

__global__ void matrixAdd(int **a,int **b,int **c){
    
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    c[i][j] = a[i][j]+b[i][j];
}

int main(){
    int N=16;
    int **a = (int **)malloc(N*N*sizeof(int));
    int **b = (int **)malloc(N*N*sizeof(int));
    int **c = (int **)malloc(N*N*sizeof(int));
    for(int i=0;i<N;i++){
        a[i]=(int*)malloc(N*sizeof(int));
        b[i]=(int*)malloc(N*sizeof(int));
        c[i]=(int*)malloc(N*sizeof(int));
        for(int j=0;j<N;j++){
            a[i][j]=1;
            b[i][j]=1;
        }
    }

    dim3 blocksPerGrid(N/16,N/16,1);
    dim3 threadsPerBlock(16,16,1);
    matrixAdd<<<blocksPerGrid,threadsPerBlock>>>(a,b,c);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<c[i][j]<<" ";
        }
        cout<<endl;
    }

}
