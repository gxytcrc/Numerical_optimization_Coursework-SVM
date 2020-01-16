function hessian = Hessian(A,alpha,t,y,C)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    N = length(alpha);
    mid = zeros(1,N);
    hessian = zeros(N+1, N+1);
    for i = 1:N
        mid(1,i) = (1/t)*((1/((alpha(i))^2)) + (1/((C-alpha(i))^2)));
    end
    
    %hessian(1:N,1:N) = A +diag((1/t)*((1/((alpha(i))^2)) +
    %(1/((C-alpha(i))^2))));
    
    hessian(1:N, 1:N) = A + diag(mid);
    for j =1:N
        hessian(N+1,j)=y(j);
        hessian(j,N+1)=y(j);
    end
end

