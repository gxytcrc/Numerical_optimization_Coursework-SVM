function [gradient] = gradient(A,alpha,lambda,t, y, C)

%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    N = length(alpha);
    gradient = zeros(N+1,1);
    
    mid = A * alpha;
    for i = 1:N
        gradient(i,1) = -1 + mid(i) + lambda*y(i)-(1/t)*(1/alpha(i) - 1/(C-alpha(i))); 
    end
    
    % gradient(1:N) = A-1+lambda*y-(1/t)*(1/alpha - 1/(C-alpha));
    gradient(N+1,1) = alpha' * y;
    
end

