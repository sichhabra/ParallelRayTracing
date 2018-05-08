#include "imager.h"
#include "antialias.h"

namespace Imager{

    Color cuda_antiAlias(double **red,double **green,double **blue,int i,int j,int antiAliasFactor){
        Color sum(0.0,0.0,0.0,1.0);
        for(int di=0;di<antiAliasFactor;di++){
            for(int dj=0;dj<antiAliasFactor;dj++){
                int x=antiAliasFactor*i + di;
                int y=antiAliasFactor*j + dj;
                Color temp(red[x][y],green[x][y],blue[x][y],1.0);
                sum += temp;
            }
        }
        return sum;
    }
}
