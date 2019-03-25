function te = MI_gaussian(Y,X)
% X: N x k_x  Y: N x k_y 
% condtional mutual information
% te = gaussian mutual information X,Y
te=0;
XY=[X Y];
kx = size(X,2);
XY_cov = cov([X Y]);
c=det(XY_cov);
if c>0
    X_cov = XY_cov(1:kx,1:kx);
    Y_cov = XY_cov(kx+1:end,kx+1:end);
    a=det(X_cov);
    b=det(Y_cov);
    te=0.5*log(a*b/c);
end
return