% intel(R) core(TM) i7- 2620M CPU @ 2.70GHz
clear;clc;
load('data_trials');
[n ntrials nvar]=size(data);
order=1;
ndmax=8;
nd=4;
file=['data_trials_' num2str(ndmax) '_' num2str(nd)];
init=false;
times=[10 50 100 150 200];
nt=length(times);
for i=1:nt
    T=times(i);
    datatot=zeros(n,T,nvar);
    datatot=data(:,1:T,:);
    tic
    [ytotc{i} indc{i}]=init_partial_conditioning_par(datatot,ndmax,order);
    time_c(i)=toc;
    tic
    pcgc_c{i}=partial_CGC_fix_nd(datatot,order,nd,indc{i});
    timefix_c(i)=toc;
    tic
    [ytotm{i} indm{i}]=init_partial_conditioning_par_m(datatot,ndmax,order);
    time_m(i)=toc;
    tic
    pcgc_m{i}=partial_CGC_fix_nd_m(datatot,order,nd,indm{i});
    timefix_m(i)=toc;
    disp([time_c(i) time_m(i) timefix_c(i) timefix_m(i)]);
end
save(file,'nvar','times','ytotc','indc','ytotm','indm','time_m','time_c','pcgc_c','pcgc_m','timefix_m','timefix_c');
