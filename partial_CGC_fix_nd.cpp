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
bool custom_isnan(double var) {
    volatile double d = var;
    return d != d;
}
void printmat(double *data, int n, int nvar) {
    int i, j, imax=12;
    if(imax>n) imax=n;
//             printf("_________________________(%d,%d)__________________\n",n,nvar);
    for(i=0;i<imax;i++){
        for(j=0; j<nvar; j++) printf("%7.4f ", data[n*j+i]);
        printf("\n");
    }
    if(n>imax){
        for(j=0; j<nvar; j++) printf("%7.4f ", data[n*j+n-1]);
        printf("\n");
    }
}
void printint(int *data, int n, int nvar) {
    int i, j, imax=12;
    for(i=0;i<n;i++){
        for(j=0; j<nvar; j++) printf("%d ", data[n*j+i]);
        printf("\n");
    }
}
double pscal(double *a, double *b, int n){
    int i;
    double ris=0.;
    for(i=0;i<n;i++) ris+=a[i]*b[i];
    return ris;
}
void prod(double *v, double *mat, int n, double *w){
    int i, j;
    for(i=0;i<n;i++) {
        w[i]=0;
        for(j=0;j<n;j++) w[i]+=v[j]*mat[n*j+i];
    }
}
void reshape(double *data, int *m, int nvar, int ntrials){
    int i, k, j, ip1, ip2, n=(*m), len=n*nvar*ntrials;
    double *work=new double[len];
    std::copy(data, data+len, work);
    for(k=0;k<nvar;k++) for(j=0;j<ntrials;j++){
        ip1=j*n+k*ntrials*n;
        ip2=j*n*nvar+k*n;
        for(i=0;i<n;i++) data[ip1+i]=work[ip2+i];
    }
    *m=ntrials*n;
    delete[] work;
}
void inverse(double* A, int n) {
    ptrdiff_t N=n, LWORK = N*N, INFO;
    ptrdiff_t *IPIV = new ptrdiff_t[N+1];
    double *WORK = new double[LWORK];
    
    dgetrf(&N, &N, A, &N, IPIV, &INFO);
    dgetri(&N, A, &N, IPIV, WORK, &LWORK, &INFO);
    
    delete IPIV;
    delete WORK;
}
void setdiff(int *v, int n, int elem) {
    int i, k=0;
//     v=std::remove(v,v+n,elem);
    for(i=0;i<n;i++) if(v[i]!=elem) v[k++]=v[i];
}
double mean(int n, double *v) {
    int i;
    double vm=0;
    for(i=0;i<n;i++) vm+=v[i];
    vm/=n;
    return vm;
}
void cov(int row, int col, double *v, double *r) {
    int i, j, k;
    double *vm=new double[col];
    for (i=0;i<col;i++) vm[i]=mean(row, &v[i*row]);
    for(i=0;i<col;i++){
//         r[i*col+i]=1;
        for(j=i;j<col;j++) {
            r[i*col+j]=0;
            for(k=0;k<row;k++) r[i*col+j]+=(v[i*row+k]-vm[i])*(v[j*row+k]-vm[j]);
            r[i*col+j]/=(row-1);
            r[j*col+i]=r[i*col+j];
        }
    }
}
double cgc_ols2(double *y, double *x, double *z, int n, int nd, int order){
    double one=1., zero=0., covX, c=0.,arg,num,den;
    int nx=1, nzx=(nd+1)*order, nzxy=nzx+order;
    int i, k, j, N=n-order, dim=nzxy+nx;
    double *C=new double[N*dim], *cov_C=new double[dim*dim];
    double *covXzxy=new double[nzxy];
    double *work=new double[nzxy];
    double *covzx=new double[nzx*nzx];
    double *covzxy=new double[nzxy*nzxy];
    for(k=0;k<order;k++){
        for(j=0;j<N;j++) C[k*N+j]=x[k+j];
        for(i=0;i<nd;i++) for(j=0;j<N;j++) C[(order*(i+1)+k)*N+j]=z[n*i+k+j];
        for(j=0;j<N;j++) C[(nzx+k)*N+j]=y[k+j];
    }
    for(j=0;j<N;j++) C[(dim-1)*N+j]=x[order+j];
    cov(N, dim, C, cov_C);
    covX=cov_C[0];
    for(i=0;i<nzx;i++){
        for(j=0;j<nzx;j++) covzx[nzx*j+i]=cov_C[dim*j+i];
    }
    for(i=0;i<nzxy;i++) {
        covXzxy[i]=cov_C[dim*(dim-1)+i];
        for(j=0;j<nzxy;j++) covzxy[nzxy*j+i]=cov_C[dim*j+i];
    }
    inverse(covzx, nzx);
    prod(covXzxy, covzx, nzx, work);
    num=covX-pscal(work, covXzxy, nzx);
    if(num <=0)return 0;
    inverse(covzxy, nzxy);
    prod(covXzxy, covzxy, nzxy, work);
    den=covX-pscal(work, covXzxy, nzxy);
    if(den <=0)return 0;
    arg=num/den;
    if(arg>1.0) c=log(arg);
    delete[] C;
    delete[] cov_C;
    delete[] covXzxy;
    delete[] covzx;
    delete[] covzxy;
    delete[] work;
    return c;
}
void partial_cgc(double *data, int n, int nvar, int nd, int order, double *ind, double *pcgc) {
    int i, j, k, drive, target, lenz,row=nvar,col=nvar;
    int *A=new int[nd+1];
    double *z=new double[n*nd];
    double *x, *y, *p;
    for(drive=0;drive<row;drive++){
        for(target=0;target<col;target++){
            if(drive!=target){
                for(i=0;i<nd+1;i++){A[i]=(int)ind[nvar*i+drive]-1;}
                setdiff(A, nd+1, target);
                y=&data[n*drive];
                x=&data[n*target];
                lenz=0;
                for(i=0;i<nd;i++){
                    p=&data[n*A[i]];
                    std::copy(p, p+n, z+lenz);
                    lenz+=n;
                }
                pcgc[target*nvar+drive]=cgc_ols2(y, x, z, n, nd, order);
            }
        }
    }
    delete[] A;
    delete[] z;
}
/* The gateway routine. */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
//declare variables
    mxArray *H0,*H1,*H2;
    const mwSize *dims;
    int i,j,k,ndmax,nd, order, ndim, n, nvar, ip=0, ntrials=1;
    double *data, *ind, *pcgc;
//Check the number of arguments
    if(nrhs != 4) mexErrMsgTxt("Wrong number of input arguments.");
    if(nlhs > 1) mexErrMsgTxt("Too many output arguments.");
// Check arguments
    ndim=(int)mxGetNumberOfDimensions(prhs[0]);
    if(ndim<2 || ndim>3) mexErrMsgTxt("data must be a 2D or 3D  full matrix.");
    if(!ISREALSCALAR(prhs[1])) mexErrMsgTxt("order must be a real double scalar.");
    if(!ISREALSCALAR(prhs[2])) mexErrMsgTxt("nd must be a real double scalar.");
    if(!ISREAL2DFULLDOUBLE(prhs[3])) mexErrMsgTxt("data must be a double full matrix.");
    //get input
    data=mxGetPr(prhs[0]);
    order=int(mxGetScalar(prhs[1]));
    nd=int(mxGetScalar(prhs[2]));
    ind=mxGetPr(prhs[3]);
    ndmax=mxGetN(prhs[3]);
//get dimensions
    dims = mxGetDimensions(prhs[0]);
    if(ndim==3) {ip=1;ntrials=(int)dims[0];}
    n=(int)dims[ip]*ntrials;
    nvar=(int)dims[ip+1];
//associate   outputs
    H0 = plhs[0] = mxCreateDoubleMatrix(nvar, nvar, mxREAL);
//associate pointer
    pcgc = mxGetPr(H0);
// do partial_cgc
    partial_cgc(data, n, nvar, nd, order, ind, pcgc);
}