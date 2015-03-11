function splitvector=generator(label,divid,times,cc)
%

%%

label = full(label);
rand('state',sum(100*label*cc));
classlabel=unique(label);
N=length(label);
presult=zeros(1,N);
for time=1:times
    perm=randperm(N);
    blewlabel=label(perm);
    for i=1:length(classlabel)
        tmp=find(blewlabel==classlabel(i));
        mita=randperm(length(tmp));
        presult(tmp)=mod(mita-ones(1,length(tmp)),divid)+ones(1,length(tmp));
    end
    for i=1:N
        splitvector(time,i)=presult(find(perm==i));
    end
end
return;
