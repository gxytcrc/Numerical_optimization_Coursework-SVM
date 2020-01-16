function [H,L] = obtain_bound(y_train,alpha,C,i,j)
    if y_train(i) == y_train(j)
        L=max(0,alpha(i)+alpha(j)-C);
        H=min(C,alpha(i)+alpha(j));
    else 
        L=max(0,alpha(j)-alpha(i));
        H=min(C,C+alpha(j)-alpha(i));
    end
end

