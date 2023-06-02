import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import callbacks
from sklearn.mixture import GaussianMixture
import time
import random
from sklearn import preprocessing as pp
from scipy import signal
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

def get_upper_tr(M):
    """
        Return upper triangle from a Square Matrix
    """
    l = M.shape[0]
    upper = int(l*(l-1)/2)
    tri = np.zeros((upper,))
    counter = 0
    for i in range(l):
        for j in range(i+1,l):
            tri[counter,] = M[j,i]
            counter += 1
    return tri

def return_nfeatures(features, n_features):
    """
        From features -> extract the center N (n_features) features

        inputs: 
            features - ndarray containing the features
            n_features - number of features to extract

        returns: 
            features - ndarray containing the center N features
    """

    return features[:, features.shape[1]//2-n_features//2:features.shape[1]//2+n_features//2]

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

def r1_minus_r3(z):
    '''
    R1-R3(NIB) Estimator

        Parameters:
                z (2darray): Time series of the observed nodes

        Returns:
                R1-R3 (2darray): Estimated connectivity matrix
    '''
    z = z.T
    tsize = z.shape[1]

    #1-lag correlation matrix
    z1=z[:,2:tsize];
    z2=z[:,1:tsize-1];
    R1=np.matmul(z1,z2.T)/(tsize-1)

    #3-lag correlation matrix
    z1=z[:,4:tsize];
    z2=z[:,1:tsize-3];
    R3=np.matmul(z1,z2.T)/(tsize-3);

    #(R1-R3)
    return (R1-R3)

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
    z1=z[:,2:tsize];
    z2=z[:,1:tsize-1];
    R1=np.matmul(z1,z2.T)/(tsize-1)

    #id - inv(R1+R2+id) Estimator
    return R1


def plot_estimator_results(As,pred,method,estimator,undirected):
    '''
    Computes the estimators' performance metrics from their estimated matrix A

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

    # Get the estimation performance metrics
    accuracy=cluster_pred(np.copy(comp[0,:]),np.copy(comp[1,:]),method,estimator)
    clvar,idgap=get_metrics(np.copy(comp[0,:]), np.copy(comp[1,:]), alln, nc)
    return np.array([accuracy,clvar,idgap])


#%% Get estimators prediction 
def get_r1_minus_r3_preds(timeseries):
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
    preds = get_upper_tr(r1_r3).reshape((-1,1))

    cl = KMeans(n_clusters=2, n_init='auto')
    cl.fit(preds)

    # norm
    centroids = cl.cluster_centers_
    if centroids[0] > centroids[1] :
        cl.cluster_centers_ = np.array([centroids[1],centroids[0]])

    preds = cl.predict(preds) + 1
    return preds

def get_r1_preds(timeseries):
    z=timeseries.T
    tsize = z.shape[1]

    # 1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)

    preds = get_upper_tr(R1).reshape((-1,1))

    cl = KMeans(n_clusters=2, n_init='auto')
    cl.fit(preds)

    # norm
    centroids = cl.cluster_centers_
    if centroids[0] > centroids[1] :
        cl.cluster_centers_ = np.array([centroids[1],centroids[0]])

    preds = cl.predict(preds) + 1
    return preds

def get_granger_preds(timeseries, method = 'gmm'):
    z=timeseries.T
    tsize = z.shape[1]

    # 0-lag correlation matrix
    R0=np.matmul(z,z.T)/tsize

    # 1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)

    # R1*inv(R0)
    r1_r0 =  np.matmul(R1,np.linalg.inv(R0))

    preds = get_upper_tr(r1_r0).reshape((-1,1))
    if method=='gmm':
        gmm = GaussianMixture(n_components=2, n_init=10)
        gmm.fit(preds)

        # norm
        centroids = gmm.means_
        if centroids[0] > centroids[1] :
            gmm.means_ = np.array([centroids[1],centroids[0]])

        preds = gmm.predict(preds) + 1
    else: 
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(preds)

        # norm
        centroids = kmeans.cluster_centers_
        if centroids[0] > centroids[1] :
            kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

        preds = kmeans.predict(preds) + 1
    return preds


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
        cld=clc
        clc=temp
        labels_unsorted[:] = np.abs(labels_unsorted[:]-1)


    # Count the number of pairs correctly classified
    tcout = 0
    for i in range(len(labels_unsorted)):
        if((labels_unsorted[i]==1 and true[i]>0) or (labels_unsorted[i]==0 and true[i]==0)):
            tcout+=1

    # Compute accuracy
    accuracy = 100-(tcout/len(labels_unsorted)*100)
    return accuracy


def compare_granger(As, z, undirected):
    '''
    This function computes the estimations (using the estimators) and their performance

        Parameters:
                As (2darray): Ground truth matrix A
                z (2darray): Matrix containing the time series of the observed nodes
                sigm (string): Clustering method to be applied
                estimator (string): Estimator type: granger, r1-r3 or r1

        Returns:
                data (1darray): Matrix containing the performance for the different 3 estimators
    '''

    #Granger
    pred = granger(z)
    metrics = plot_estimator_results(As,pred,"gmm",'G',undirected)


    return metrics


def compare_estimators(As,z,undirected):
    '''
    This function computes the estimations (using the estimators) and their performance

        Parameters:
                As (2darray): Ground truth matrix A
                z (2darray): Matrix containing the time series of the observed nodes
                sigm (string): Clustering method to be applied
                estimator (string): Estimator type: granger, r1-r3 or r1

        Returns:
                data (1darray): Matrix containing the performance for the different 3 estimators
    '''
    #Granger
    pred = granger(z)
    metrics1 = plot_estimator_results(As,pred,"gmm",'G',undirected)

    #R1 minus R3
    pred = r1_minus_r3(z)
    metrics2 = plot_estimator_results(As,pred,'gmm','R13',undirected)

    #R1 Estimator
    pred = r1(z)
    metrics3 = plot_estimator_results(As,pred,'gmm','R12',undirected)

    data = np.zeros((3,3))
    data[0,:] = metrics1
    data[1,:] = metrics2
    data[2,:] = metrics3
    return data

def get_metrics(true, pred, upper, nc):
    '''
    This function rescales the data and computes the performance metrics, this way the measurements are calculated using the
    same scale

        Parameters:
                true (1darray): Ground truth matrix A
                pred (1darray): Estimated matrix A
                upper (int): Number of pairs of nodes
                nc (int): Number of connected pairs

        Returns:
                idgap (int): Identifiability gap
                clvar (int): Cluster Variance
    '''
    #Normalize the values
    pred = sklearn.preprocessing.minmax_scale(pred, feature_range=(0, 1), axis=0, copy=True)

    #Initialize the structures to save the data
    con = np.zeros((nc))
    dis = np.zeros((upper-nc))

    #Split the values belonging to connected and disconnected pairs
    c2,c3 = 0,0
    for i in range(len(true)):
        if(true[i]>0):
            con[c2] = pred[i]
            c2+=1
        else:
            dis[c3] = pred[i]
            c3+=1

    #Threshold
    mint = np.max(dis)
    maxt = np.min(con)
    threshold = np.ones((upper,))*((mint+maxt)/2)

    #Compute metrics
    clvar = (np.var(dis)+np.var(con))/2
    idgap = maxt - ((mint+maxt)/2)

    #If the id gap is negative then we cant correctly classify the pairs as disconnected or connected
    if(idgap < 0):
        idgap=0

    return clvar,idgap


def extract_cross_correlation(A,zz,n):
    '''
    Extracts the cross correlation from the feature

        Parameters:
                zz (1darray): Matrix with the observed node time series
                n (int): Number of nodes in the graph
        Returns:
                data_nn (2darray): Matrix containing a fecture vector for each pair of nodes
                y (1darray): Ground-truth
    '''
    tsize = zz.shape[0]
    upper = int(n*(n-1)/2)
    data_nn = np.zeros((200,upper))
    target = np.zeros((1,upper))

    counter = 0

    #Go through each pair and compute the time laged cross-correlation
    for j in range(n):
        for k in range(j+1,n):
            aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
            data_nn[:,counter] = aux[tsize-100:tsize+100]
            target[0,counter] = A[j,k]
            counter = counter + 1

    data_nn = data_nn/np.max(data_nn)
    data_nn=data_nn.T

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y


def extract_cross_correlation_features(A,zz,n, n_features):
    '''
    Extracts the cross correlation from the feature

        Parameters:
                zz (1darray): Matrix with the observed node time series
                n (int): Number of nodes in the graph
        Returns:
                data_nn (2darray): Matrix containing a fecture vector for each pair of nodes
                y (1darray): Ground-truth
    '''
    tsize = zz.shape[0]
    upper = int(n*(n-1)/2)
    data_nn = np.zeros((n_features,upper))
    target = np.zeros((1,upper))
    offset = n_features//2
    counter = 0

    #Go through each pair and compute the time laged cross-correlation
    for j in range(n):
        for k in range(j+1,n):
            aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
            data_nn[:,counter] = aux[tsize-offset:tsize+offset]
            target[0,counter] = A[j,k]
            counter = counter + 1

    data_nn = data_nn/np.max(data_nn)
    data_nn=data_nn.T

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y


    
def extract_cross_correlation_ss(A,zz,n):
    '''
    Extracts the cross correlation from the feature.
    With standard scaling

        Parameters:
                zz (1darray): Matrix with the observed node time series
                n (int): Number of nodes in the graph
        Returns:
                data_nn (2darray): Matrix containing a fecture vector for each pair of nodes
                y (1darray): Ground-truth
    '''
    tsize = zz.shape[0]
    upper = int(n*(n-1)/2)
    data_nn = np.zeros((200,upper))
    target = np.zeros((1,upper))

    counter = 0

    #Go through each pair and compute the time laged cross-correlation
    for j in range(n):
        for k in range(j+1,n):
            aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
            data_nn[:,counter] = aux[tsize-100:tsize+100]
            target[0,counter] = A[j,k]
            counter = counter + 1

    data_nn=data_nn.T
    scaler = StandardScaler()
    data_nn = scaler.fit_transform(data_nn)

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y

  
def extract_cross_correlation_ss_features(A,zz,n,n_features):
    '''
    Extracts the cross correlation from the feature.
    With standard scaling

        Parameters:
                zz (1darray): Matrix with the observed node time series
                n (int): Number of nodes in the graph
                n_features (int): size of features
        Returns:
                data_nn (2darray): Matrix containing a fecture vector for each pair of nodes
                y (1darray): Ground-truth
    '''
    tsize = zz.shape[0]
    upper = int(n*(n-1)/2)
    data_nn = np.zeros((n_features,upper))
    target = np.zeros((1,upper))
    counter = 0 

    offset = int(n_features/2)

    #Go through each pair and compute the time laged cross-correlation
    for j in range(n):
        for k in range(j+1,n):
            aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
            data_nn[:,counter] = aux[tsize-offset:tsize+offset]
            target[0,counter] = A[j,k]
            counter = counter + 1

    data_nn=data_nn.T
    scaler = StandardScaler()
    data_nn = scaler.fit_transform(data_nn)

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y


def extract_cross_correlation_unscaled_features(A,zz,n,n_features):

    '''
    Extracts the cross correlation from the feature

        Parameters:
                zz (1darray): Matrix with the observed node time series
                n (int): Number of nodes in the graph
        Returns:
                data_nn (2darray): Matrix containing a fecture vector for each pair of nodes
                y (1darray): Ground-truth
    '''
    tsize = zz.shape[0]
    upper = int(n*(n-1)/2)
    data_nn = np.zeros((n_features,upper))
    target = np.zeros((1,upper))

    counter = 0
    offset = n_features//2

    #Go through each pair and compute the time laged cross-correlation
    for j in range(n):
        for k in range(j+1,n):
            aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
            data_nn[:,counter] = aux[tsize-offset:tsize+offset]
            target[0,counter] = A[j,k]
            counter = counter + 1

    data_nn=data_nn.T

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y
    
def extract_cross_correlation_unscaled(A,zz,n):
    '''
    Extracts the cross correlation from the feature

        Parameters:
                zz (1darray): Matrix with the observed node time series
                n (int): Number of nodes in the graph
        Returns:
                data_nn (2darray): Matrix containing a fecture vector for each pair of nodes
                y (1darray): Ground-truth
    '''
    tsize = zz.shape[0]
    upper = int(n*(n-1)/2)
    data_nn = np.zeros((200,upper))
    target = np.zeros((1,upper))

    counter = 0

    #Go through each pair and compute the time laged cross-correlation
    for j in range(n):
        for k in range(j+1,n):
            aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
            data_nn[:,counter] = aux[tsize-100:tsize+100]
            target[0,counter] = A[j,k]
            counter = counter + 1

    data_nn=data_nn.T

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y

def plot_svm_results(model_name,data_nn,y):
    '''
    Loads a saved SVM model and classfies the features into connected or disconnected classes

        Parameters:
                model_name (string): Name of the saved model
                data_nn (1darray): Features of the observed models
                y (1darray): Ground-truth
        Returns:
                [per,clvar,idgap] (1darray): Performance of the SVM predictions
    '''
    loaded_model = pickle.load(open(model_name, 'rb'))
    pred = loaded_model.predict(data_nn)
    true = y.flatten()

    #Compute the metrics
    upper = int((se-ss)*((se-ss)-1)/2)
    tr = np.triu(As)
    np.fill_diagonal(tr,0)
    tr = tr>0
    tr=tr.astype(int)
    nc = np.sum(tr)
    clvar,idgap=get_metrics(np.copy(true-1), np.copy(pred), upper, nc)

    per=100-(sum(pred!=true)/len(pred)*100)

    return [per,clvar,idgap]

def load_model(name):
    '''
    Loads a saved SVM model

        Parameters:
                name (string): Name of the saved model
        Returns:
                model (keras.object):SVM Model
    '''
    return keras.models.load_model(name)

def plot_nn_results(model_name,data_nn, y , se , ss, As):
    '''
    Loads a saved CNN model and classfies the features into connected or disconnected classes

        Parameters:
                model_name (string): Name of the saved model
                data_nn (1darray): Features of the observed models
                y (1darray): Ground-truth
        Returns:
                [accuracy_nn,clvar,idgap] (1darray): Performance of the CNN predictions
    '''
    #Loads the models
    model = load_model(model_name)

    #Divide the connected and disconnected pairs values
    idd = y < 2
    idd = np.squeeze(idd)

    idc = y > 1
    idc = np.squeeze(idc)

    #Reshape the data to 3 dimensions
    data_nn = data_nn.reshape((data_nn.shape[0], data_nn.shape[1], 1))

    #Predicts the weight of the connection and classifies the data
    dis_mpred = model.predict(data_nn[idd,:,:])
    truec_nn = np.sum(dis_mpred<1.5)
    con_mpred = model.predict(data_nn[idc,:,:])
    trued_nn = np.sum(con_mpred>1.5)

    tall = truec_nn + trued_nn

    accuracy_nn = tall/(len(dis_mpred)+len(con_mpred))*100

    #Compute the metrics
    upper = int((se-ss)*((se-ss)-1)/2)
    tr = np.triu(As)
    np.fill_diagonal(tr,0)
    tr = tr>0
    tr=tr.astype(int)
    nc = np.sum(tr)
    clvar,idgap=get_metrics(np.copy(y-1), np.copy(model.predict(data_nn)), upper, nc)
    return [accuracy_nn,clvar,idgap]

def save_pickles(name,comparison_data,sz_main,p):
    '''
    Saves the performance metrics of all estimators

        Parameters:
                comparison_data (2darray): Performance of different estimators over a range of number of samples
                sz_main (int): Number of observed nodes
                p (1darray): Probability of the Erdős–Rényi Graph Model
    '''
    output = open(name, 'wb')
    pickle.dump(comparison_data, output)
    output.close()

def save_pickle(name,comparison_data):
    '''
    Saves the performance metrics of all estimators

        Parameters:
                comparison_data (2darray): Performance of different estimators over a range of number of samples

    '''
    output = open(name, 'wb')
    pickle.dump(comparison_data, output)
    output.close()

def get_inverted_features(timeseries, n=200):
    z = timeseries.T  
    tsize = z.shape[1]
    sz = z.shape[0]
    upper = int(sz*(sz-1)/2)

    features = np.zeros((upper,n*2))
    counter=0
    for offset in range(n):
        z1 = z[:,offset+1:tsize]
        z2 = z[:,1:tsize-offset]
        R = np.matmul(z1,z2.T)/(tsize-offset)
        r_inv = get_upper_tr(np.linalg.inv(R))
        r = get_upper_tr(R)
        features[:,counter] = r
        features[:,counter+1] = r_inv
        counter += 2
    return features

def filter_timeseries(timeseries, order=2, fs=1.38, cutoff_high=0.15, cutoff_low=0.01):
    from scipy.signal import butter, lfilter, freqz


    def butter_lowpass(cutoff, fs, order=5, btype='low'):
        return butter(order, cutoff, fs=fs, btype=btype, analog=False)

    def butter_lowpass_filter(data, cutoff, fs, order=5, btype='low'):
        b, a = butter_lowpass(cutoff, fs, order=order, btype=btype)
        y = lfilter(b, a, data)
        return y

    timeseries_filtered = np.zeros(timeseries.shape)
    for x in range(timeseries_filtered.shape[1]):
        # Demonstrate the use of the filter.
        # First make some data to be filtered.
        sig = timeseries[:,x]

        # Filter the data, and plot both the original and filtered signals.
        y = butter_lowpass_filter(sig, cutoff_high, fs, order)
        y = butter_lowpass_filter(y, cutoff_low, fs, order, btype='high')
        scaler = StandardScaler()
        y = scaler.fit_transform(y.reshape(-1,1))

        timeseries_filtered[:,x] = y.ravel()

    return timeseries_filtered  