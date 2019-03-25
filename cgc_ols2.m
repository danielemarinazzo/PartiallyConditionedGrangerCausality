function [cgc] = cgc_ols2(y,x,z,order,nd)
[N,nx]=size(x);
C=zeros(N-order,order*(nd+2)+1);
for k = 1:order
    C(:,k) = x(k : N-order+k-1) ;
    for i=1:nd
        C(:,order*i+k) = z(k : N-order+k-1,i) ;
    end
    C(:,(nd+1)*order+k) = y(k : N-order+k-1) ;
end
C(:,(nd+2)*order+1)=x(order+1:end);
cc=cov(C);
nzx=(nd+1)*order;
nzxy=(nd+2)*order;
covX=cc(1,1);
covzx=cc(1:nzx,1:nzx);
covzxy=cc(1:nzxy,1:nzxy);
covXzx=cc(1+nzxy,1:nzx);
covXzxy=cc(1+nzxy,1:nzxy);

covXzx = covX - covXzx/covzx*covXzx';
covXzxy = covX - covXzxy/covzxy*covXzxy';
cgc = covXzx/covXzxy;
if cgc>0
    cgc=log(cgc);
else
    cgc=0;
end