#Required packages to run the code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import EarlyStopping
from sklearn.mixture import GaussianMixture
from keras.models import *
from keras.layers import *
import sklearn
from sklearn import svm
import pickle
import geopandas
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm.notebook import trange
from statistics import mean
import warnings
warnings.filterwarnings("ignore")


#Save or load a trainned CNN model
def save_model_cnn(model,name):
    model.save(name)

def load_model_cnn(name):
    return keras.models.load_model(name)

def save_model_svm(name,model):
    pickle.dump(model, open(name, 'wb'))
    
def load_model_svm(name):
    return pickle.load(open(name, 'rb'))

def get_adjacency(sz,p,undirected):
    '''
    Generates a realization of an Erdős–Rényi Graph Model, undirected or directed.
    -First generates of matrix of random floating point numbers in the range [0.0, 1.0].
    -If those values are <=p then there is no edge between pairs
    -Makes the matrix symmetric if the graoh is undirected

        Parameters:
                sz (int): Number of nodes
                p (int): Probability of existing an edge between each pair of nodes

        Returns:
                adj (2darray): Adjacency matrix
    '''
    adj = np.random.random((sz, sz)) <= p
    adj = np.triu(adj.astype(int))
    np.fill_diagonal(adj,0)
    if(undirected):
        adj = adj + adj.T
    return adj

def get_A(adj,c,rho):
    '''
    Generates the connectivity matrix (interaction weights) from the adjacency matrix according to the laplacian rule

        Parameters:
                adj (2darray): Adjacency matrix
                c,rho (int): Numbers between 0 and 1, to make the spectral radius < 1

        Returns:
                A (2darray): Connectivity matrix
    '''    
    sz = len(adj)
    Dvec = np.sum(adj, axis=1)
    Dmax = np.max(Dvec)
    ccc = c*1/Dmax
    D = np.diag(Dvec)
    L = D - adj
    Ap = np.eye(sz) - ccc*L
    A = rho * Ap
    return A

def gen_linear_time_series(A,tsize,x0,noise):
    '''
    Generates the syntetic time series data given the connectivity matrix and the initial condiction x(0), 
    according to the dynnamical rule y(n + 1) = Ay(n) + x(n + 1) + alpha/n*1*1^t X'(n+1)

        Parameters:
                A (2darray): Connectivity matrix
                tsize (int): Time series size - number of samples
                x0 (int): Initial condition x(0), in this case is zero
                noise (2darray): Noise matrix

        Returns:
                x (2darray): Time series data of the graph
    ''' 
    sz = len(A)
    x = np.zeros((tsize,sz))
    x[0,:] = np.ones((1,sz))*x0
    
    for i in range(1,tsize):
        x[i,:] = np.dot(A,x[i-1,:]) + noise[i,:]
    
    return x

def generate_noise(N, n_samples, alpha, beta):
    '''
        y(n + 1) = alpha * x1(n+1) + beta * 1*1.T * X2(n+1)

        Parameters:
                N (int): number of nodes
                n_samples (int): number of samples
                alpha (float): Standard Deviation of noise X1
                Beta (float): Standard Deviation of noise X2

        Returns:
                z (2darray): Time series data of the graph
    ''' 
    
    ones = np.ones((N,N)) * beta/np.sqrt(N)
    
    z = np.zeros((n_samples, N))
    
    for i in range(n_samples):
        x1 = np.random.normal(size=(1,N))
        x2 = np.random.normal(size=(1,N))
        
        z[i,:] = alpha * x1 + np.matmul(x2,ones)
        
    return z

def granger(z):
    '''
    Granger Estimator

        Parameters:
                z (2darray): Time series of the observed nodes

        Returns:
                R1*inv(R0) (2darray): Estimated connectivity matrix
    '''   
    z=z.T
    tsize = z.shape[1]
    
    #0-lag correlation matrix
    R0=np.matmul(z,z.T)/tsize

    #1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)

    #R1*inv(R0)
    return np.matmul(R1,np.linalg.inv(R0))

def get_r1_minus_r3_preds(timeseries,undirected):
    z=timeseries.T
    tsize = z.shape[1]

    #1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)

    #3-lag correlation matrix
    z1=z[:,4:tsize]
    z2=z[:,1:tsize-3]
    R3=np.matmul(z1,z2.T)/(tsize-3)

    r1_r3 = R1 - R3

    return R1 - R3

def r1(z):
    '''
    R1 Estimator

        Parameters:
                z (2darray): Time series of the observed nodes

        Returns:
                R1 (2darray): Estimated connectivity matrix
    '''
    z = z.T
    sz = z.shape[0]
    tsize = z.shape[1]
    
    #1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)

    #id - inv(R1+R2+id) Estimator
    return R1

def get_upper_tr(M,undirected):
    """
        Return upper triangle from a Square Matrix
    """
    l = M.shape[0]

    if(undirected):
        upper = int(l*(l-1)/2)
        tri = np.zeros((upper,))
        counter = 0
        for i in range(l):
            for j in range(i+1,l):
                tri[counter,] = M[j,i]
                counter += 1
    else:
        dsize = (l*l)-l
        tri = np.zeros((dsize,))
        counter = 0
        for i in range(l):
            for j in range(l):
                if(i != j):
                    tri[counter,] = M[j,i]
                    counter += 1
    return tri

def create_dataset(sz,tsize,undirected,A,time_series,n_features):
    '''
    Generates the synthectic data, extracts the features and returns the tranning/testing dataset

        Parameters:
                sz (int): Number of nodes
                p (int): Probability of existing an edge between each pair of nodes
                c,rho (int): Numbers between 0 and 1, to make the spectral radius < 1  
                tsize (int): Time series size - number of samples
                x0 (int): Initial condition x(0), in this case is zero
                qsi (int): Noise standart deviation 
                A (2darray): Grown-Truth matrix A

        Returns:
                data (2darray): Matrix containing the feature-vectors between each pair of nodes
                target (1darray): Ground-truth - pairs are connected or disconnected
    '''     
    nFeatures = int(n_features/2)

    #Is the graph undirected or directed
    if(undirected):
        
        #Create data structures
        upper = int(sz*(sz-1)/2)  #Number of elements in the upper matrix
        data = np.zeros((n_features,upper))
        target = np.zeros((1,upper))
        
        #Goes through each pair (of the upper matrix) and computes the time laged cross-correlation (excludes diagonal)
        counter = 0
        for j in range(sz):
            for k in range(j+1,sz):
                #Compute the cross correlation
                aux = signal.correlate(time_series[:,j],time_series[:,k], mode="full")
                #Extracts the first negative and positive lags
                data[:,counter] = aux[tsize-nFeatures:tsize+nFeatures]
                #Saves the data
                target[0,counter] = A[j,k]
                counter = counter + 1
    else:
        #Create data structures
        dsize = (sz*sz)-sz       #Number of elements excluding the diagonal
        data = np.zeros((200,dsize))
        target = np.zeros((1,dsize))
        
        #Goes through each pair and computes the time laged cross-correlation (excludes diagonal)
        counter = 0
        for j in range(sz):
            for k in range(sz):
                if(j!=k):
                    #Computes the cross correlation
                    aux = signal.correlate(time_series[:,j],time_series[:,k], mode="full")
                    #Extracts the firs negative and positive lags
                    data[:,counter] = aux[tsize-nFeatures:tsize+nFeatures]
                    #Saves the data
                    target[0,counter] = A[j,k]
                    counter = counter + 1 
    return data,target

def get_inverted_features(timeseries, n, undirected):
    z = timeseries.T  
    tsize = z.shape[1]
    sz = z.shape[0]

    if (undirected):

        upper = int(sz*(sz-1)/2)

        features = np.zeros((upper,n*2))
        counter = 0
        for offset in range(n):
            z1 = z[:,offset+1:tsize]
            z2 = z[:,1:tsize-offset]
            R = np.matmul(z1,z2.T)/(tsize-offset)
            r_inv = get_upper_tr(np.linalg.inv(R), undirected)
            r = get_upper_tr(R,undirected)
            features[:,counter] = r
            features[:,counter+1] = r_inv
            counter += 2

    else:
        dsize = (sz*sz)-sz
        features = np.zeros((dsize,n*2))
        counter = 0
        for offset in range(n):
            z1 = z[:,offset+1:tsize]
            z2 = z[:,1:tsize-offset]
            R = np.matmul(z1,z2.T)/(tsize-offset)
            r_inv = get_upper_tr(np.linalg.inv(R),undirected)
            r = get_upper_tr(R,undirected)
            features[:,counter] = r
            features[:,counter+1] = r_inv
            counter += 2

    return features

def cluster_pred(true,pred,method,estimator):
    '''
    Clusters the estimators' results into two groups and classifies it into connected and disconnected

        Parameters:
                true (2darray): Ground truth matrix A
                pred (2darray): Estimated matrix A
                method (string): Clustering method to be applied
                estimator (string): Estimator type: granger, r1-r3 or r1

        Returns:
                accuracy (1darray): Accuracy of the estimations
    '''

    #Sort it
    idx = np.argsort(pred, axis=0)

    #Build a data structure for clustering
    features = np.zeros((len(pred),2))
    features[:,0] = np.linspace(0,1,len(pred))
    features[:,1] = pred[idx]
    
    #Clustering data with Gaussian Mixture Model or kmeans
    if(method=="gmm"):
        gmm = GaussianMixture(n_components=2)
        labels = gmm.fit_predict(features)
    elif(method=="kmeans"):
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(features)

    #Unsort the values after clustering
    us = np.argsort(idx, axis=0)
    labels_unsorted = labels[us]
    cld = pred[np.where(labels_unsorted == 1)]
    clc = pred[np.where(labels_unsorted == 0)]

    #Disconnected pairs must be on cld and conneceed pairs on clc
    if(np.mean(cld)>np.mean(clc)):
        temp = cld
        cld = clc
        clc = temp
        labels_unsorted[:] = np.abs(labels_unsorted[:]-1)
        
    
    #Count the number of pairs correctly classified
    tcout = 0
    for i in range(len(labels_unsorted)):
        if((labels_unsorted[i]==1 and true[i]>0) or (labels_unsorted[i]==0 and true[i]==0)):
            tcout+=1
        
    #Compute accuracy
    accuracy = 100-(tcout/len(labels_unsorted)*100)
    return accuracy

def estimator_results(As,pred,method,estimator,undirected):
    '''
    Computes the estimators' performance metrics from their estimated matrix A or nn models preds

        Parameters:
                As (2darray): Ground truth matrix A
                pred (2darray): Estimated matrix A
                method (string): Clustering method to be applied
                estimator (string): Estimator type: granger, r1-r3 or r1

        Returns:
                R1 (1darray): Performance metrics (accuracy, identifiability gap)
    '''
    
    #Number of nodes
    sz = len(As)

    #Count the number of connected and disconnected pairs
    if(undirected):
        alln = int(sz*(sz-1)/2)
        tr = np.triu(As)
        np.fill_diagonal(tr,0)
        tr = tr>0
        tr=tr.astype(int)
        nc = np.sum(tr)
    else:
        alln = (sz*sz)-sz
        tr = As
        np.fill_diagonal(tr,0)
        tr = tr>0
        tr=tr.astype(int)
        nc = np.sum(tr)
        
    #Divide the pairs into connected and disconnected
    con = np.zeros((nc))
    dis = np.zeros((alln-nc))
    comp = np.zeros((2,alln))

    #Save data
    c1,c2,c3 = 0,0,0
    if(undirected):
        for i in range(sz):
            for j in range(i+1,sz):
                comp[0,c1] = As[i,j]
                comp[1,c1] = pred[i,j]
                c1=c1+1
                if(As[i,j]>0):
                    con[c2] = pred[i,j]
                    c2+=1
                else:
                    dis[c3] = pred[i,j]
                    c3+=1
    else:
        for i in range(sz):
            for j in range(sz):
                if(j!=i):
                    comp[0,c1] = As[i,j]
                    comp[1,c1] = pred[i,j]
                    c1=c1+1
                    if(As[i,j]>0):
                        con[c2] = pred[i,j]
                        c2+=1
                    else:
                        dis[c3] = pred[i,j]
                        c3+=1
                
    #Get the estimation performance metrics
    accuracy=cluster_pred(np.copy(comp[0,:]),np.copy(comp[1,:]),method,estimator)
    #clvar,idgap=get_metrics(np.copy(comp[0,:]), np.copy(comp[1,:]), alln, nc)
    #return {"accuracy":accuracy,"clvar":clvar,"idgap":idgap}
    return {"accuracy":accuracy}

def shift_features(colored_features,lowest_feature):
    sz1,sz2 = colored_features.shape
    
    new_features = np.zeros((sz1,sz2))
    
    for i in range(sz2):
        new_features[:,i] = colored_features[:,i] - lowest_feature

    return new_features

def normalize_features(features, tsize):
    """
        Normalize the features based on the number of terms of correlate
    """
    offset = features.shape[1] // 2
    return features / np.arange(tsize-offset, tsize+offset)

def shift_centroid(data,centroid):

    sz1,sz2 = data.shape

    new_data = np.zeros((sz1,sz2))

    for i in range(sz2):
        new_data[:,i] = data[:,i] - centroid[i]

    return new_data

#Parameters
t_samples = 2000001
step = 5000
n_runs = 10
observable_n = 10

sz = 100    #Number of nodes
#ps = [0.20,0.50,0.70,0.90]  #Probability of nodes being connected (Erdős–Rényi)
ps = [0.2,0.4,0.7,0.9]
c = 0.9
rho = 0.75
undirected = False

#Define the range of noise variance
alpha = 1
#betas = np.arange(0.1,1.1,0.1)
#betas = [1,5,10]
beta = 10
x0 = 0            #Initial condition

x = np.arange(step,t_samples,step)

s1 = "./models/Linear/diagonal_100nodes_cnn"
s2 = "_200_newfeatures_0.50p"

aux_models = ["_sergio","_max","_ssmax","_ss","_ssss"]

models = []

for i in aux_models:
    models.append(s1 + i + s2)

aux_models.pop(0)

s1 = "./models/Linear/diagonal_100nodes_ff"
s2 = "_200_newfeatures_0.50p"

for i in aux_models:
    models.append(s1 + i + s2)

betas_performance_list = np.zeros((len(ps),len(models)+3,len(x)))

n_prob = 0

for p in ps:
    print("Prob: " + str(p))

    count_2 = 0

    models_performance = np.zeros((len(models)+3,len(x)))

    for samples in x:
        print("Samples: " + str(samples))
        performance_list = np.zeros((len(models)+3,n_runs))

        for i in range(n_runs):

            adj = get_adjacency(sz,p,undirected)

            A = get_A(adj,c,rho)

            noise = generate_noise(sz,samples,alpha,beta)

            #get time series with diagonal noise
            time_series = gen_linear_time_series(A,samples,x0,noise)

            sc = StandardScaler()
            time_series_ss = sc.fit_transform(time_series) 

            new_A = A[0:observable_n,0:observable_n]
            new_time_series = time_series[:,0:observable_n]
            new_time_series_ss = time_series_ss[:,0:observable_n]

            #Granger
            A_granger = granger(new_time_series)
            acc_granger = estimator_results(new_A,A_granger,'gmm','granger',undirected)['accuracy']

            performance_list[0][i] = acc_granger

            #R1 - R3
            r1_r3 = get_r1_minus_r3_preds(new_time_series,undirected)
            acc_r1_r3 = estimator_results(new_A,r1_r3,'kmeans','r1_r3',undirected)['accuracy']

            performance_list[1][i] = acc_r1_r3

            #R1
            r1_preds = r1(new_time_series)
            acc_r1 = estimator_results(new_A,r1_preds,'kmeans','r1',undirected)['accuracy']

            performance_list[2][i] = acc_r1

            data_sergio,target = create_dataset(observable_n,samples,undirected,new_A,new_time_series,200)

            data_scaled = get_inverted_features(new_time_series,100,undirected)
            data_scaled_ss = get_inverted_features(new_time_series_ss,100,undirected)

            data_sergio = data_sergio/np.max(data_sergio)

            data_max = data_scaled/np.max(data_scaled)
            data_ssmax = data_scaled_ss/np.max(data_scaled_ss)

            sc = StandardScaler()
            data_ss = sc.fit_transform(data_scaled)
            sc = StandardScaler()
            data_ssss = sc.fit_transform(data_scaled_ss)

            #datas = [data_cnn_50,data_cnn_100,data_cnn_150,data_cnn_200,data_cnn_250,data_cnn_300,data_cnn_350,data_cnn_400,data_cnn_450,data_cnn_500]
            datas = [data_sergio.T,data_max,data_ssmax,data_ss,data_ssss,data_max,data_ssmax,data_ss,data_ssss]

            #get predicts
            count = 0
            for m in models:
                model = load_model_cnn(m)

                pred = model.predict(datas[count],verbose = 0)

                acc = cluster_pred(target.flatten(),pred.flatten(),'kmeans','cnn')

                performance_list[count+3][i] = acc

                count += 1

        #save granger
        models_performance[0][count_2] = np.mean(performance_list[0])
        models_performance[1][count_2] = np.mean(performance_list[1])
        models_performance[2][count_2] = np.mean(performance_list[2])

        for j in range(len(models)):
            models_performance[j+3][count_2] = np.mean(performance_list[j+3])
        
        count_2 += 1

    betas_performance_list[n_prob,] = models_performance

    n_prob += 1

probs_newfeatures_directed_100nodes_beta10 = betas_performance_list

with open('probs_newfeatures_directed_100nodes_beta10.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(probs_newfeatures_directed_100nodes_beta10, file)