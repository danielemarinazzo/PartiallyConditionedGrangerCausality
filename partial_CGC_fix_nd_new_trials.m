function pcgc=partial_CGC_fix_nd_new_trials(datatot,order,nd,ind)
% once you have found nd, the number of variables for the partial
% conditioning set, you can compute the partially conditioned granger
% causality
[trials,nvar,N] = size(datatot);
data=zeros(nvar,N*trials);
for it=1:trials
    data(:,(it-1)*N+1:it*N)=squeeze(datatot(it,:,:));
end
data=data';

pcgc=zeros(nvar,nvar);
parfor drive=1:nvar
    for target=1:nvar
%         disp(num2str([drive target]))
        if drive ~= target
            A=ind(drive,:);
            conz = A(~ismembc(A(:), target));
            conz = conz(1:nd);
            pcgc(drive,target) = cgc_ols2(data(:,drive),data(:,target),data(:,conz),order,trials); %this uses covariance matrix. you can use your favorite TE estimator here
        end
    end
end