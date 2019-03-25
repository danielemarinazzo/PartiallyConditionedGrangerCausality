#include <matrix.h>
#include <mex.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif
#define ISREAL2DFULLDOUBLE(P) (!mxIsComplex(P) && mxGetNumberOfDimensions(P) == 2 && !mxIsSparse(P) && mxIsDouble(P))
#define ISREALSQUAREMATRIXDOUBLE(P) (ISREAL2DFULLDOUBLE(P) && mxGetN(P) == mxGetM(P))
#define ISREALROWVECTORDOUBLE(P) (ISREAL2DFULLDOUBLE(P) && mxGetM(P) == 1)
#define ISREALCOLUMNVECTORDOUBLE(P) (ISREAL2DFULLDOUBLE(P) && mxGetN(P) == 1)
#define ISREALSCALAR(P) (ISREAL2DFULLDOUBLE(P) && mxGetNumberOfElements(P) == 1)
#if !defined(_WIN32)
#define dgemm dgemm_
#define dgeev dgeev_
#define dsyev ssyev_
#endif
#include "lapack.h"
#include "blas.h"
#include <iostream>     // std::cout
#include <algorithm>    // std::copy
#include <vector>       // std::vector

/* computational subroutine */

void printmat(double *data,int n,int nvar) {
    int i, j,imax=10;
    if(imax>n) imax=n;
            printf("_________________________(%d,%d)__________________\n",n,nvar);
    for(i=0;i<imax;i++){
//         for(j=0; j<nvar; j++) printf("%7.4f ", data[n*j+i]);
        for(j=0; j<nvar; j++) printf("%e ", data[n*j+i]);
        printf("\n");
    }
    if(n>0){
        for(j=0; j<nvar; j++) printf("%7.4f ", data[n*j+n-1]);
        printf("\n");
    }
}
double Determinant(double* A, int n) {
    int i;
    double det=1;
    ptrdiff_t N=n, LWORK = N*N, INFO;
    ptrdiff_t *IPIV = new ptrdiff_t[N+1];
    dgetrf(&N, &N, A, &N, IPIV, &INFO);
    for(i=0;i<n;i++) det=det*A[i*n+i];
    delete[] IPIV;
    return abs(det);
}
    
// double Determinant(double* mat, int n){
//     int i;
//     double det=1;
//     if(n==1) det=mat[0];
//     else if(n==2) det=mat[0]*mat[3]-mat[1]*mat[2];
//     else {
//         ptrdiff_t la=n, lwork=3*n-1, info;
//         double *D=new double[n],*work=new double[lwork];
//         dsyev("N", "U", &la, mat, &la, D, work, &lwork, &info);
//         for(i=0;i<n;i++) det=det*D[i];
//         delete[] D;
//         delete[] work;
//     }
//     return det;
// }
void meanstd(int n, double *v, double *vm, double *sd) {
    int i;
    double v2=0;
    *vm=0;
    for(i=0;i<n;i++) {*vm+=v[i];v2+=v[i]*v[i];}
    v2/=n;
    *vm/=n;
    *sd=sqrt((v2-(*vm)*(*vm))*n/(n-1));
}
void cov(int row, int col, double *v, double *r) {
    int i, j, k;
    for(i=0;i<col;i++){
        r[i*col+i]=1;
        for(j=i+1;j<col;j++) {
            r[i*col+j]=0;
            for(k=0;k<row;k++) r[i*col+j]+=v[i*row+k]*v[j*row+k];
            r[i*col+j]/=(row-1);
            r[j*col+i]=r[i*col+j];
        }
    }
}
void zscore(double *data, int N, int order, double *Y) {
    int i, j;
    double xm, sd;
    for(i=0;i<order;i++) {
        meanstd(N, &data[N*i], &xm, &sd);
        for(j=0;j<N;j++) Y[N*i+j]=(data[N*i+j]-xm)/sd;
    }
}
double MI_gaussian(double *Y,double *X,int N,int ky,int kx){
    int i, j, k=kx+ky;
    double px, py, pxy,te=0.;
    double *XY=new double[N*k];
    double *xy_cov=new double[k*k];
    double *x_cov=new double[kx*kx], *y_cov=new double[ky*ky];
    std::copy(X, X+N*kx, XY);
    std::copy(Y, Y+N*ky, XY+N*kx);
    cov(N, k, XY, xy_cov);
    for(i=0;i<kx;i++) for(j=0;j<kx;j++) x_cov[kx*j+i]=xy_cov[k*j+i];
    for(i=0;i<ky;i++) for(j=0;j<ky;j++) y_cov[ky*j+i]=xy_cov[k*(kx+i)+(kx+j)];
    pxy=Determinant(xy_cov, k);
    if(pxy>0) {px=Determinant(x_cov, kx);py=Determinant(y_cov, ky);te=0.5*log(px*py/pxy);}
    delete[] x_cov;
    delete[] y_cov;
    delete[] XY;
    delete[] xy_cov;
    return te;
}
void infogain(int drive, double *X, int nvar, int ndmax, int N, int order, double *y, double *ind){
    int i,k,nd,id,nx=N*order,n1=nvar, len_Zt=0,kmax;
    double *t,*ti,*tdrive,z,zmax=-1000.;
    int *indici=new int[n1];
    double *Zt=new double[nx*ndmax];
    t=&X[nx*drive];
    k=0;for(i=0;i<nvar;i++) if(i!=drive) indici[k++]=i;
    for(nd=0;nd<ndmax;nd++){
        n1--;
        for(k=0;k<n1;k++){
            ti=&X[nx*indici[k]];
            double *Zd=new double[nx+len_Zt];
            std::copy(Zt, Zt+len_Zt, Zd);
            std::copy(ti, ti+nx, Zd+len_Zt);
            z=MI_gaussian(Zd,t,N,(nd+1)*order,order);
            if (z>zmax){zmax=z;kmax=k;}
            delete[] Zd;
        }
        id=indici[kmax];
        y[nvar*nd+drive]=zmax;
        ind[nvar*nd+drive]=id+1;
        tdrive=&X[N*order*id];
        std::copy(tdrive, tdrive+nx, Zt+len_Zt);
        len_Zt=len_Zt+nx;
        std::remove(indici,indici+n1,id);
    }
    delete[] Zt;
    delete[] indici;
}
void Init(double *data, int n, int nvar, int ndmax, int order, double *y, double *ind) {
    int i, j, k, N=n-order, drive,nx=N*order;
    double *past=new double[nx], *X=new double[nx*nvar];
    for(j=0;j<nvar;j++){
        for(i=0;i<N;i++) for(k=0;k<order;k++) past[k*N+i]=data[j*n+k+i];
        zscore(past,N,order,&X[j*nx]);
    }
    for(drive=0;drive<nvar;drive++) infogain(drive, X, nvar, ndmax, N, order,y,ind);
    delete[] past;
    delete[] X;
}
/* The gateway routine. */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
//declare variables
    mxArray *H0, *H1, *H2;
    const mwSize *dims;
    int ndmax=10, order=1, n, nvar, N, i, j, ky,ndim,ip=0,ntrials=1;
    double *data, *y, *ind, *X, vm, sd, *v2, te;
//Check the number of arguments
    if(nrhs != 3) mexErrMsgTxt("Wrong number of input arguments.");
    if(nlhs > 2) mexErrMsgTxt("Too many output arguments.");
// Check arguments
    ndim=(int)mxGetNumberOfDimensions(prhs[0]);
    if(ndim<2 || ndim>3) mexErrMsgTxt("data must be a 2D or 3D  full matrix.");
    if(!ISREALSCALAR(prhs[1])) mexErrMsgTxt("ndmax must be a real double scalar.");
    if(!ISREALSCALAR(prhs[2])) mexErrMsgTxt("order must be a real double scalar.");
    //get input
    data=mxGetPr(prhs[0]);
    ndmax=int(mxGetScalar(prhs[1]));
    order=int(mxGetScalar(prhs[2]));
//get dimensions
    dims = mxGetDimensions(prhs[0]);
    if(ndim==3) {ip=1;ntrials=(int)dims[0];}
    n=(int)dims[ip]*ntrials;
    nvar=(int)dims[ip+1];
//associate   outputs
    H0 = plhs[0] = mxCreateDoubleMatrix(nvar, ndmax, mxREAL);
    H1 = plhs[1] = mxCreateDoubleMatrix(nvar, ndmax, mxREAL);
//associate pointer
    y = mxGetPr(H0);
    ind = mxGetPr(H1);
//do init
    Init(data, n, nvar, ndmax, order, y, ind);
}