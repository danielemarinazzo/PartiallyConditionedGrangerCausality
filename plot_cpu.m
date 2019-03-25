clear;clc;
ndmax=6;
nd=3;
file=['data_trials_' num2str(ndmax) '_' num2str(nd)];
load(file);
nt=length(times);
for i=1:nt 
    dind(i)=sum(sum(indm{i}-indc{i}));
    dy(i)=sum(sum(abs(ytotm{i}-ytotc{i})));
    dpcgc(i)=sum(sum(abs(pcgc_m{i}-pcgc_c{i})));
end
disp(sprintf('%d %7.6f %7.6f',sum(dind),sum(dy),sum(dpcgc)));
fs=12;
figure(1);clf;plot(100*times,time_c,'-*r',100*times,time_m,'-b*');
set(gca,'Fontsize',fs);
xlabel('N','Fontsize',fs);
ylabel('cpu time (sec)','Fontsize',fs)
legend('cpp','matlab');
title(['init ndmax=' num2str(ndmax) ' nvar=' num2str(nvar)]);
figure(2);clf;plot(100*times,timefix_c,'-*r',100*times,timefix_m,'-b*');
set(gca,'Fontsize',fs);
xlabel('N','Fontsize',fs);
ylabel('cpu time (sec)','Fontsize',fs)
legend('cpp','matlab');
title(['partial nd=' num2str(nd) ' nvar=' num2str(nvar)]);

