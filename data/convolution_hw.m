
clear
close all
clc

load convolution_data

pl = length(pulse);           % get the length of wavelet
rl = length(reflectivity);    % get the length of reflectivity

c = conv(reflectivity,pulse);

d1 = dftmtx(pl);
d2 = dftmtx(rl);

figure;
p_freq = d1*pulse;
plot(abs(d1*pulse));

figure;
plot(c);


% From the assignment powerpoint, we can find the following principal:
% c(0) = f(1)*g(1);
% c(1) = f(1)*g(2) + f(2)*g(1);
% c(2) = f(1)*g(3) + f(2)*g(2) + f(3)*g(1);
% .....
% c(n) = f(1)*g(n+1) + f(2)*g(n) +...+ f(n+1)*g(1);
% .....
% c(m+n-2) = f(m)*g(n);
