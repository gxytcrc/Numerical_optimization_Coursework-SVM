function [svm] = SVM_train(X_train,Y_train,C, k_type, M_type)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
[K] = kernel(k_type,X_train, X_train); 
[rows,cols] = size(Y_train);
Alpha0 = 0.05*ones(rows,1);

switch M_type
    case'interior'
        A = (Y_train * Y_train') .* K;
        [alpha, lambda, solution,Iter] = interior_point(Alpha0, A, Y_train, C);
        
        epsilon = 1e-4;
        sv_lable = find(abs(alpha)>epsilon);
        svm.X_sv = X_train(sv_lable,:);
        svm.Y_sv = Y_train(sv_lable);
        svm.alpha = alpha;
        svm.alpha_sv = alpha(sv_lable);
        mid = (svm.alpha_sv'.*svm.Y_sv')*kernel(k_type,svm.X_sv,svm.X_sv);
        svm.b = mean(svm.Y_sv' - mid);
        
        svm.solution = solution;
        svm.inner = Iter.inner;
        svm.outer = Iter.outer;
    case'smo'
        [Alpha,smo] = SMO(Alpha0,X_train,Y_train, C,k_type);
        epsilon = 1e-4;
        sv_lable = find(abs(Alpha)>epsilon);
        svm.X_sv = smo.x(sv_lable,:);
        svm.Y_sv = smo.y(sv_lable);
        svm.alpha = Alpha;
        svm.alpha_sv = Alpha(sv_lable);
        mid = (svm.alpha_sv'.*svm.Y_sv')*kernel(k_type,svm.X_sv,svm.X_sv);
        svm.b = mean(svm.Y_sv' - mid);
        
        svm.outer = smo.history;

end

