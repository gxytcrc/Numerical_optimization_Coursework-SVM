function [alpha,smo] = SMO(alpha0,x_train,y_train, C,kernel_type)
eps = 1e-4;
alpha = alpha0;
K = kernel(kernel_type,x_train,x_train);
N=length(y_train);
%generate the alpha history, used to reduece the computaion time
alpha_history(:,1) = alpha0 - 10000;

for Iter = 1:1000
    for i=1:N
        for j=1:N
            [H,L] = obtain_bound(y_train,alpha,C,i,j);
            E1=sum(alpha.*y_train.*K(:,i))-y_train(i);
            E2=sum(alpha.*y_train.*K(:,j))-y_train(j);
            data0 = 2*K(i, j) - K(i,i)^2-K(j,j)^2;
            if data0>=0
                continue
            end
            
            %update alpha
            alpha_old=alpha(j);
            data2 = -( y_train(j)*(E1-E2) )/data0; 
            alpha(j)=alpha(j)+data2;
            if alpha(j) > H
                alpha(j) = H;
            end
            if alpha(j) < L
                alpha(j) = L;
            end
            
            data1 = y_train(i)*y_train(j)*(alpha_old-alpha(j));
            alpha(i)=alpha(i) + data1;
        end
    end
    alpha_history(:,Iter+1) = alpha;
    diff = norm(alpha_history(:,Iter+1) - alpha_history(:,Iter));
    if Iter >1
        smo.history(Iter-1) = diff;
    end
    if norm(diff)<eps
        break
    end
end
smo.x = x_train;
smo.y = y_train;
end

