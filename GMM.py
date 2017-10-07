from numpy import sum, prod, mean, size, shape, reshape, where, zeros, ones, matrix, eye, mat
from math import exp, sqrt, pi, log
from scipy.linalg import norm, det, pinv
from skimage import io
from sklearn.cluster import KMeans
from time import clock
 
start = clock()

def Gauss(Data, Mu, Sigma):
       
#Author:	Louis, 2017
#
#Inputs -----------------------------------------------------------------
#  o Data:  D x N matrix representing N datapoints of D dimensions.
#  o Mu:    D x K matrix representing the centers of the K GMM components.
#  o Sigma: D x D x K matrix representing the Covariance matrices of the 
#           K GMM components.
#Outputs ----------------------------------------------------------------
#  o prob:  1 x N matrix representing the probabilities for the 
#           N datapoints.    

       D = shape(Data)
       D = D[0]
       Data = Data-Mu
       power = -0.5*(Data.T@pinv(Sigma)@Data)
       prob = exp(power)/sqrt((2*pi)**D*(abs(det(Sigma))+1))
       return prob

       
path='C:\Zephyr\Python\*.jpg'
Image = io.ImageCollection(path)
(N,x,y,z)=shape(Image)
Image = reshape(Image,(N,x*y,z))
D = x*y


# Notation
#m,n=size of image m×n=D
#N=num of image
#I0=Mu1(j,:) Finally
#Epsilon=threshold
#Alpha=Pi or weight
#Beta=Switch for N and N+U
#Mu1(j,:) Mu2(j,:)
#Sigma1 Sigma2
#P1=posterior probability of right images
#P2=1-P1 posterior probability of wrong images
#Time=Total iteration
#L_old=previous loglikely
#K=2 Clusters


# Initialization
Image_R = Image[:,:,0]
Image_R = reshape(Image_R,(N,D))
K = KMeans(n_clusters=2).fit(Image_R)
Index = K.labels_
Center = K.cluster_centers_
Mu = zeros(shape=[2,D,3])
Prior = zeros(2)

for i in range(2):
       Location = matrix(where(Index==i)).T   
       Prior[i] = size(Location)
       Mu[i,:,:]= sum(Image[Location,:,:],axis=0)

P1 = Index
P2 = ones(N)-P1
Mu1 = Mu[0,:,:]/Prior[0]
Mu2 = Mu[1,:,:]/Prior[1]
Cov1 = zeros(shape=[D,3,3])
Cov2 = zeros(shape=[D,3,3])
Num1 = zeros(shape=[N,3,3])
Num2 = zeros(shape=[N,3,3])
Num3 = zeros(shape=[N,D,3])
Num4 = zeros(shape=[N,D,3])

for i in range(N):
    for j in range(D):
        Cov1[j,:,:] = mat(Image[i,j,:]-Mu1[j,:]).T*mat(Image[i,j,:]-Mu1[j,:])
        Cov2[j,:,:] = mat(Image[i,j,:]-Mu2[j,:]).T*mat(Image[i,j,:]-Mu2[j,:])      
    MCov1 = mean(Cov1,axis=0)
    MCov2 = mean(Cov2,axis=0)
    Num1[i,:,:] = P1[i]*MCov1
    Num2[i,:,:] = P2[i]*MCov2
Sigma1 = sum(Num1,axis=0)/sum(P1)+eye(3)
Sigma2 = sum(Num2,axis=0)/sum(P2)+eye(3)

Prior = len(Location)/N
Alpha = 1-Prior
Beta = 0


# Parameters Setting
Epsilon = 1e-4
L_old = 0
Good = zeros(D)
Bad = zeros(D)

# Uniform
dist = zeros(N*(N+1)//2) # note the // here to make it int
k = 0
for i in range(N):
    for j in range(N-i):
        dist[0] = norm(Image[i,:,:]-Image[j,:,:])
        k = k+1
        
distance = max(dist)
U = 1/(distance+1)


# Starting GMM

for w in range(1):  

    # Estimation
    for i in range(N):   
        for j in range(D):
            #  recalculate posterior probability
            Good[j] = Alpha*Gauss(Image[i,j,:],Mu1[j,:],Sigma1)
            Bad[j] = (1-Alpha)*(Beta*Gauss(Image[i,j,:],Mu2[j,:],Sigma2)+(1-Beta)*U)
        P1[i] = sum(Good)/sum(Good+Bad)
    P2 = ones(N)-P1
    
    
    # Maximization
    
    for i in range(N):
        Num3[i,:,:] = P1[i]*Image[i]
        Num4[i,:,:] = P2[i]*Image[i]
    Mu1 = sum(Num3,axis=0)/sum(P1)
    Mu2 = sum(Num4,axis=0)/sum(P2)
    
    for i in range(N):
           for j in range(D):
                  Cov1[j,:,:] = mat(Image[i,j,:]-Mu1[j,:]).T*mat(Image[i,j,:]-Mu1[j,:])
                  Cov2[j,:,:] = mat(Image[i,j,:]-Mu2[j,:]).T*mat(Image[i,j,:]-Mu2[j,:])        
           MCov1 = mean(Cov1,axis=0)
           MCov2 = mean(Cov2,axis=0)
           Num1[i,:,:] = P1[i]*MCov1
           Num2[i,:,:] = P2[i]*MCov2
    Sigma1 = sum(Num1,axis=0)/sum(P1)
    Sigma2 = sum(Num1,axis=0)/sum(P2)   

    Alpha = sum(P1)/N
    
    
    # Check Likelihood
    
    # Test β=0
    
    Beta = 0   
    
    for i in range(N):   
        for j in range(D):
            #  recalculate posterior probability
            Good[j] = Alpha*Gauss(Image[i,j,:],Mu1[j,:],Sigma1)
            Bad[j] = (1-Alpha)*(Beta*Gauss(Image[i,j,:],Mu2[j,:],Sigma2)+(1-Beta)*U)
    Good = Good**(1/D)
    Good = prod(Good)
    Bad = Bad**(1/D)
    Bad = prod(Bad)
    
    for i in range(N):
        L0 = P1[i]*log(Good)+P2[i]*log(prod(Bad))
      
    L0 = sum(L0)


    # Test β=1
    
    Beta = 1
    
    for i in range(N):   
        for j in range(D):
            #  recalculate posterior probability
            Good[j] = Alpha*Gauss(Image[i,j,:],Mu1[j,:],Sigma1)
            Bad[j] = (1-Alpha)*(Beta*Gauss(Image[i,j,:],Mu2[j,:],Sigma2)+(1-Beta)*U)
    
    G = Good**(1/D)
    G = prod(Good)
    B = Bad**(1/D)
    B = prod(Bad)
    
    for i in range(N):
        L1 = P1[i]*log(prod(G))+P2[i]*log(prod(B))   
    
    L1 = sum(L1)
    
    if L0 < L1:
        L = L0
        Beta = 0  
    
    if L0 > L1:
        L = L1
        Beta = 1  
    
    if abs((L/L_old)-1) < Epsilon:
        break
    else:
        L_old = L
    

# Output Synthetic Image

I0 = reshape(Mu1,x,y,3)
io.show(I0)

end = clock()
print("Time used:", end-start)

