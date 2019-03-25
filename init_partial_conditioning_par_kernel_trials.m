function [y ind]=init_partial_conditioning_par_kernel_trials(datatot,ndmax,order,sigmax,sigmay)
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
[N,nvar] = size(data);
n=N/trials;
X=cell(nvar,1);
past_ind = repmat([1:order],n-order,1) + repmat([0:n-order-1]',1,order);
for i=1:nvar
    past_data=[];
for j=1:trials
%now
past_data_c=reshape(data((j-1)*n+past_ind,i),n-order,order);

%%%%%
%%%ora accumulo
past_data=[past_data;past_data_c];
end
    X{i}=zscore(past_data);
end

ind=zeros(nvar,ndmax);
y=ind;
% now you call the info_gain function for each candidate driver
%sigmax=10;sigmay=10;
parfor drive=1:nvar
    %tic
    [y(drive,:) ind(drive,:)]=info_gain_kernel(drive,X,nvar,ndmax,sigmax,sigmay);
    %toc
    %pause
end

%when you have finished, you can plot the increment of y vs nd to see where
%to stop

% you can adopt other strategies, i.e. increment below a certain threshold
% etc, but I am quite happy for the visual
figure;plot(1:ndmax-1,diff(y'));


