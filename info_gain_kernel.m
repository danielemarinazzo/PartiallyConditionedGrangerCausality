function [y ind]=info_gain_kernel(drive,X,nvar,ndmax,sigmax,sigmay)
indici=setdiff(1:nvar,drive); %eliminate the candidate driver from the set
t=X{drive};
Zt=[];
% params.sigx=-1;
% params.sigy=-1;
params.sigx=sigmax;
params.sigy=sigmay;
for nd=1:ndmax
    n1=length(indici);
    z=zeros(n1,1);
    for k=1:n1
        Zd=[Zt X{indici(k)}];
        z(k) = hsicTestGamma_noThr(Zd,t,params);
        %z(k)= MI_gaussian(Zd,t); %compute conditional MI, here from covariance matrix, but you can use the exact formula
    end
    [y(1,nd) id]=max(z); %greedy algorithm, find the max contribution, store it and remove it from the set of candidates
    Zt=[Zt X{indici(id)}];
    ind(1,nd)=indici(id);
    indici=setdiff(indici,indici(id));
end