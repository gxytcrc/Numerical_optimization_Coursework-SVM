function [alpha, lambda,solution,Iter] = interior_point(alpha0, A, y, C)
    t = 1;
    lambda = 1;
    alpha = alpha0;
    %Newton method
    solution(:,1) = alpha;
    eps_inner = 1*10^-8;
    eps_outer = 1*10^-8;
    count = 1;
    for n = 1:200
        for i = 1:100
            H = Hessian(A, alpha, t, y, C);
            G = gradient(A, alpha, lambda, t, y, C);
            searchdir = (H\G);
            adir = searchdir(1:end-1);

            alpha = alpha - adir;
            lambda = lambda - searchdir(end);
            
            Iter.inner(count)=norm(alpha-alpha0);
            count = count +1;
            if norm(alpha-alpha0)<eps_inner
                break;
            end
            alpha0 = alpha;
        end
        t = t*1.08;
        solution(:,n+1) = alpha;
        if norm(solution(:,n+1)-solution(:,n))<eps_outer
            break;
        end
        Iter.outer(n) =  norm(solution(:,n+1)-solution(:,n));
    end
   
end

