clear all;
%if the code excuation is fail, please run again
n = 30;

%generate the x1 (class 1) data, 
%if you want try the new data, please comment the line 8, uncoment line 7
%x1 = randn(n,2);
load('analysis_linear_x1.mat');
%load('analysis_non_linear_x1.mat');
y1 = ones(n,1); 

%generate the x2 (class 2 )data
%if you want try the new data, please comment the line 15, uncoment line 14
%x2 = 3+randn(n,2);
load('analysis_linear_x2.mat');
%load('analysis_non_linear_x2.mat');
y2 = -ones(n,1);

figure; plot(x1(:,1),x1(:,2), 'bs')
hold on; plot(x2(:,1), x2(:,2),'r+');

%slack parameter
C = 1;

%combine two classes data for training
X_train = [x1;x2];
Y_train = [y1;y2];
%training process
[svm] = SVM_train(X_train,Y_train,C,'linear','smo');
plot(svm.X_sv(:,1),svm.X_sv(:,2),'ro');


%% test process
[x_test1,x_test2] = meshgrid(-7:0.05:7,-7:0.05:7);

[rows,cols] = size(x_test1);
nt = rows*cols;
X_1 = reshape(x_test1,1,nt)';
X_2 = reshape(x_test2,1,nt)';

%obtain the test X
Xt = [X_1,X_2];

%plot interior point method
w = (svm.alpha_sv'.*svm.Y_sv')*kernel('linear',svm.X_sv,Xt);
Y_est = sign(w + svm.b);
Yd = reshape(Y_est,rows,cols);
contour(x_test1,x_test2,Yd,[0,0],'ShowText','off');

title('Classficition result'); xlabel('X'); ylabel('Y');   

 

%% plot converge plot

% figure;
% plot(svm.outer);
% xlabel('iteration times')
% ylabel('alpha_{k+1} - alpha_k')
% title('SMO conerage rate plot');



