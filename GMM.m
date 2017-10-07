clc
close all
clear all
 
tic
 
% Import
photo = 'C:\Zephyr\MATLAB\*.jpg';
photopath  = 'C:\Zephyr\MATLAB\';
file = dir(photo);
 
for i = 1:length(file)
    Image{i} = imread([photopath, file(i).name]);
    Image_R(:,:,i) = Image{i}(:,:,1);
end
 
[x,y,z] = size(Image_R);
Image_R = reshape(Image_R,[x*y,z]);
Image_R = im2double(Image_R);
[~,N] = size(Image);
 
[x,y,z] = size(Image{1});
for i=1:N
    Image{i} = im2double(Image{i});
    Image{i} = reshape(Image{i},[x*y,z]);
end
 
 
% Notation
% m,n=size of image m??n=D
% N=num of image
% I0=Mu1(j,:) Finally
% Epsilon=threshold
% Alpha=Pi or weight
% Beta=Switch for N and N+U
% Mu1(j,:) Mu2(j,:)
% Sigma1 Sigma2
% P1=posterior probability of right images
% P0=1-P1 posterior probability of wrong images
% Time=Total iteration
% L_old=previous loglikely
% K=2 Clusters
 
% Initialization
Data = Image_R';
K=2; % Clusters
[N, D] = size(Data);
[Index,Center] = kmeans(Data,K);
 
for i = 1:K
    Mu(:,:,i) = zeros(x*y,z);
    Location = find(Index==i);
    Prior(i) = length(Location);
    for j=1:Prior(i)
        Mu(:,:,i) = Mu(:,:,i) + Image{Location(j)};
    end
end
 
P1 = zeros(N,1);
P1(Location) = 1;
P0 = ones(N,1)-P1;
Mu1 = Mu(:,:,1)/Prior(1);
Mu2 = Mu(:,:,2)/Prior(2);
COV1 = cell(1,D);
COV2 = cell(1,D);
 
for i = 1:N
    for j = 1:D
        COV1{j} = (Image{i}(j,:)-Mu1(j,:))'*(Image{i}(j,:)-Mu1(j,:));
        COV2{j} = (Image{i}(j,:)-Mu2(j,:))'*(Image{i}(j,:)-Mu2(j,:));
    end
    TCOV1 = cat(3,COV1{:});
    TCOV2 = cat(3,COV2{:});
    MCOV1 = mean(TCOV1,3);
    MCOV2 = mean(TCOV2,3);
    Num1(:,:,i) = P1(i)*MCOV1;
    Num2(:,:,i) = P0(i)*MCOV2;
end
Sigma1 = sum(Num1,3)/sum(P1)+eye(3);
Sigma2 = sum(Num2,3)/sum(P0)+eye(3);
 
Prior = Prior./sum(Prior);
Alpha = Prior(1);
Beta = 0;
 
 
% Parameters Setting
Epsilon = 1e-4;
L_old = -realmax;
 
k = 1;
for i = 1:N
    for j = 1:N-i
        dist(k) = norm(Image{i}-Image{j});
        k = k+1;
    end
end
 
 
% Uniform
distance = max(dist);
U = 1/(distance+1);
 
 
% Starting GMM
 
for w = 1:7
    
    % Estimation
    
    for i = 1:N
        for j = 1:D
            %  recalculate posterior probability
            Good(j) = Alpha*Gauss(Image{i}(j,:),Mu1(j,:),Sigma1);
            Bad(j) = (1-Alpha)*(Beta*Gauss(Image{i}(j,:),Mu2(j,:),Sigma2)+(1-Beta)*U);
        end
        P1(i) = sum(Good)/sum(Good+Bad);
    end
    P0 = ones(N,1)-P1;
    
    % Maximization
    
    for i = 1:N
        Num3(:,:,i) = P1(i)*Image{i};
        Num4(:,:,i) = P0(i)*Image{i};
    end
    Mu1 = sum(Num3,3)/sum(P1);
    Mu2 = sum(Num4,3)/sum(P0);
    
    
    for i = 1:N
        for j = 1:D
            COV1{j} = (Image{i}(j,:)-Mu1(j,:))'*(Image{i}(j,:)-Mu1(j,:));
            COV2{j} = (Image{i}(j,:)-Mu1(j,:))'*(Image{i}(j,:)-Mu1(j,:));
        end
        TCOV1 = cat(3,COV1{:});
        TCOV2 = cat(3,COV2{:});
        MCOV1 = mean(TCOV1,3);
        MCOV2 = mean(TCOV2,3);
        Num1(:,:,i) = P1(i)*MCOV1;
        Num2(:,:,i) = P0(i)*MCOV2;
    end
    Sigma1 = sum(Num1,3)/sum(P1)+eye(3);
    Sigma2 = sum(Num2,3)/sum(P0)+eye(3);
    Alpha = sum(P1)/N;
    
    
    % Check Likelihood
    
    % Test ??=0
    
    Beta = 0;
    
    for i=1:N
        for j=1:D
            %  recalculate posterior probability
            Good(j) = Alpha*Gauss(Image{i}(j,:),Mu1(j,:),Sigma1);
            Bad(j) = (1-Alpha)*(Beta*Gauss(Image{i}(j,:),Mu2(j,:),Sigma2)+(1-Beta)*U);
        end
    end
    
    Good = Good.^(1/D);
    Good = prod(Good);
    Bad = Bad.^(1/D);
    Bad = prod(Bad);
    
    
    for i = 1:N
        L0 = P1(i)*log(Good)+P0(i)*log(prod(Bad));
    end
    
    L0 = sum(L0);
    
    
    % Test ??=1
    
    Beta = 1;
    
    for i = 1:N
        for j = 1:D
            %  recalculate posterior probability
            Good(j) = Alpha*Gauss(Image{i}(j,:),Mu1(j,:),Sigma1);
            Bad(j) = (1-Alpha)*(Beta*Gauss(Image{i}(j,:),Mu2(j,:),Sigma2)+(1-Beta)*U);
        end
    end
    
    Good = Good.^(1/D);
    Good = prod(Good);
    Bad = Bad.^(1/D);
    Bad = prod(Bad);
    
    for i = 1:N
        L1 = P1(i)*log(prod(Good))+P0(i)*log(prod(Bad));
    end
    
    L1 = sum(L1);
    
    if L0 < L1
        L = L0;
        Beta = 0;
    end
    
    if L0 > L1
        L = L1;
        Beta = 1;
    end
    
    if abs((L/L_old)-1) < Epsilon
        break;
    else
        L_old = L;
    end
end
 
 
% Output Synthetic Image
 
I0 = reshape(Mu1,x,y,z);
imagesc(I0);
 
toc
