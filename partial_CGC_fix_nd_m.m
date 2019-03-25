function pcgc=partial_CGC_fix_nd_m(data,order,nd,ind)
% once you have found nd, the number of variables for the partial
% conditioning set, you can compute the partially conditioned granger
% causality
dims=ndims(data);
if dims==3
    [n,trials,nvar]=size(data);
    N=n*trials;
    data=reshape(data,N,nvar);
elseif dims==2 
    [~,nvar]=size(data);
else
    disp('wrong dimensions number!!!!!!!!!!!!!!!')
    return
end
pcgc=zeros(nvar,nvar);
parfor drive=1:nvar
    for target=1:nvar
        if drive ~= target
            A=ind(drive,:);
            conz = A(~ismembc(A(:), target));
            conz = conz(1:nd);
            %this uses covariance matrix. you can use your favorite TE estimator here
            pcgc(drive,target) = cgc_ols2(data(:,drive),data(:,target),data(:,conz),order,nd); 
        end
    end
end