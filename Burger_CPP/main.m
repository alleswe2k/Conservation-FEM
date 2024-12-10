clc;
clear all;

d = load("res.m");
h = sqrt(1./d(:,1));
L1 = d(:,2);
L2 = d(:,3);

ratio = h(2:end)./h(1:end-1);
p1 = log(L1(2:end)./L1(1:end-1))./log(ratio(1:end))
p2 = log(L2(2:end)./L2(1:end-1))./log(ratio(1:end))

  %loglog(h, L1, h, h.^2, h, L2, h, h.^2, 'r--')
