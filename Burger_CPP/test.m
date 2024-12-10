clc;
clear all;
%close all;
clf;

x = 0:0.01:1;
y0 = 3*x.^2 - 2*x.^3;

alpha0 = 0.5;
q = 4.0;
psi = max(0.0, (x - alpha0)/(1.0 - alpha0));
psi2 = psi.*psi;

y2 = 6*x.^5-15*x.^4 +10*x.^3; 

y = psi2;
plot(x,y0,x,y, x,y2,'r')

k = 20;
C = 0.7;
yy = 1./(1 + exp(-k*(x-C)));
hold on
plot(x,yy,'blue')