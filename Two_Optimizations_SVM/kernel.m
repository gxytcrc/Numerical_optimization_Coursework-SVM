function [K] = kernel(kernel_type,X1, X2)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
switch kernel_type
    case 'Gaussian'
        [raw,~] = size(X1);
        [raw1,~] =size(X2);
        number = sum(X1'.^2);
        number1 = sum(X2'.^2);
        K = exp((1/2^5)*(-number'*ones(1,raw1)-ones(raw,1)*number1 + 2*X1*X2'));
    case 'linear'
        K = X1*X2';
end

