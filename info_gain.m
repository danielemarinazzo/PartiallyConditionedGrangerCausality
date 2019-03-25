function [y ind]=info_gain(drive,X,nvar,ndmax)
indici=setdiff(1:nvar,drive); %eliminate the candidate driver from the set
t=X{drive};
% t=X(:,:,drive);
Zt=[];
for nd=1:ndmax
    n1=length(indici);
    z=zeros(n1,1);
    for k=1:n1
        Zd=[Zt X{indici(k)}];
%         disp(Zd(1:10,:));
%         disp(Zd(end,:));
        z(k)= MI_gaussian(Zd,t); %compute conditional MI, here from covariance matrix, but you can use the exact formula
%         disp(sprintf('drive=%d nd=%d k=%d z=%7.4f',drive,nd,k,z(k)))
    end
    [y(1,nd) id]=max(z); %greedy algorithm, find the max contribution, store it and remove it from the set of candidates
%     disp(sprintf('id=%d zmax=%7.4f',indici(id),y(nd)));
    Zt=[Zt X{indici(id)}];
    ind(1,nd)=indici(id);
    indici=setdiff(indici,indici(id));
end