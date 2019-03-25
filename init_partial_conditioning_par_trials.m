function [y ind X]=init_partial_conditioning_par_trials(datatot,ndmax,order)
% This is the first function to run
% It computes the curve of information gain for ndmax variables.
% ndmax can be max equal to nvar-1, but it's worth to stop early (a small portions of the variables)
% since it's time consuming. if no clear minimum is reached you can go further.



[trials,nvar,N] = size(datatot);
data=zeros(nvar,N*trials);
for it=1:trials
    data(:,(it-1)*N+1:it*N)=squeeze(datatot(it,:,:));
end
data=data';
[N nvar]=size(data);
disp(size(data))
X=cell(nvar,1);
for i=1:nvar
    past_data=zeros(N-order,order);
    for k = 1:order
        past_data(:,k) = data(k : N-order+k-1, i) ;
    end
    X{i}=zscore(past_data);
end

ind=zeros(nvar,ndmax);
y=ind;
% now you call the info_gain function for each candidate driver
parfor drive=1:nvar
    [y(drive,:) ind(drive,:)]=info_gain(drive,X,nvar,ndmax);
end

%when you have finished, you can plot the increment of y vs nd to see where
%to stop

% you can adopt other strategies, i.e. increment below a certain threshold
% etc, but I am quite happy for the visual
figure;plot(1:ndmax-1,diff(y'));


