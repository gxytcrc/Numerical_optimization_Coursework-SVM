clear all;
n=30;
% x1=randn(n,2);
load('analysis_muti_x1.mat');
% x2=5+randn(n,2);
load('analysis_muti_x2.mat');
% x3=randn(n,2);
% x3=[x3(:,1)+5, x3(:,2)-5];
load('analysis_muti_x3.mat');

figure;
plot(x1(:,1),x1(:,2),'bs',x2(:,1),x2(:,2),'k+',x3(:,1),x3(:,2),'r*');
hold on;
X_12=[x1;x2];
X_13=[x1;x3];
X_23=[x2;x3];
Y_all=[ones(n,1);-ones(n,1)];
C=1;
svm_12=SVM_train(X_12,Y_all,C,'linear','smo');
svm_13=SVM_train(X_13,Y_all,C,'linear','smo');
svm_23=SVM_train(X_23,Y_all,C,'linear','smo');

%test the data
[x_test1,x_test2] = meshgrid(-8:0.05:8,-8:0.05:8);
[rows,cols] = size(x_test1);
nt = rows*cols;
Xt = [reshape(x_test1,1,nt)',reshape(x_test2,1,nt)'];

%classify 1 and 2
w_12 = (svm_12.alpha_sv'.*svm_12.Y_sv')*kernel('linear',svm_12.X_sv,Xt);
result_12 = w_12 + svm_12.b;
Y_est12 = sign(result_12);
Yd_12 = reshape(Y_est12,rows,cols);


%classify 1 and 3
w_13 = (svm_13.alpha_sv'.*svm_13.Y_sv')*kernel('linear',svm_13.X_sv,Xt);
result_13 = w_13 + svm_13.b;
Y_est13 = sign(result_13);
Yd_13 = reshape(Y_est13,rows,cols);


%classify 2 and 3
w_23 = (svm_23.alpha_sv'.*svm_23.Y_sv')*kernel('linear',svm_23.X_sv,Xt);
result_23 = w_23 + svm_23.b;
Y_est23 = sign(result_23);
Yd_23 = reshape(Y_est23,rows,cols);

contour(x_test1,x_test2,Yd_13,[0,0],'ShowText','off');
contour(x_test1,x_test2,Yd_12,[0,0],'ShowText','off');
contour(x_test1,x_test2,Yd_23,[0,0],'ShowText','off');


