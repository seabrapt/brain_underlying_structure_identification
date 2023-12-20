"""
    python file that contains all general functions like:
        - Generating a graph
        - Generating the Interaction matrix
        - Generating noise and timeseries

"""

# Imports
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans 
from scipy import signal

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
    if (undirected):
        adj = np.triu(adj.astype(int))
    else: 
        adj = adj.astype(int)


    np.fill_diagonal(adj,0)
    if(undirected):
        adj = adj + adj.T


    return adj


def get_A(adj,c,rho):
    '''
    Generates the connectivity matrix (interaction weights) from the adjacency matrix according to the laplacian rule

        Parameters:
                adj (2darray): Adjacency matrix
                c, rho (int): Numbers between 0 and 1, to make the spectral radius < 1

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

def get_target(A, undirected=True):
    """
        From the interaction matrix return the structure of the graph in the form of a vector

        input: 
            A: (2darray) interaction matrix

        output:
            y: (1darray) structural connectivity
    """
    n = A.shape[0]

    if undirected==False:
        upper = int((n*n) - n)
        target = np.zeros((1,upper))
        counter = 0

        #Go through each pair and compute the time laged cross-correlation
        for j in range(n):
            for k in range(j,n):
                if j!=k:
                    target[0,counter] = A[j,k]
                    counter = counter + 1

        y=target>0
        y=y.astype(int)+1

        y=y.T
        return y
    else:
        upper = int(n*(n-1)/2)
        target = np.zeros((1,upper))
        counter = 0

        #Go through each pair and compute the time laged cross-correlation
        for j in range(n):
            for k in range(j+1,n):
                target[0,counter] = A[j,k]
                counter = counter + 1

        y=target>0
        y=y.astype(int)+1

        y=y.T
        return y


def generate_timeseries(A,tsize,x0,noise):
    '''
    Generates the syntetic time series data given the connectivity matrix and the initial condiction x(0),
    according to the dynnamical rule y(n + 1) = Ay(n) + x(n + 1)

        Parameters:
                A (2darray): Connectivity matrix
                tsize (int): Time series size - number of samples
                x0 (int): Initial condition x(0), in this case is zero
                noise (2darray) : Noise matriz

        Returns:
                x (2darray): Time series data of the graph
    '''

    sz = len(A)
    x = np.zeros((tsize,sz))
    x[0,:] = np.ones((1,sz))*x0


    for i in range(1,tsize):
        x[i,:] = np.dot(A,x[i-1,:]) + noise[i,:]
    return x

def generate_noise(N, n_samples, alpha, beta, perturbation=False):
    """
        Z(n+1) = alpha * X1(n+1) + beta * 1*1.T * X2(n+1)
        
        Inputs:
            N (int) - number of nodes
            n_samples (int) - number of samples
            alpha (float) - Standard deviation of noise X1
            beta (float)  - Standar deviation of noise X2
            
        Output: 
            Z (2darray) - Noise samples
    """
    
    ones = np.ones((N,N)) * beta/np.sqrt(N)
    if perturbation: 
        pert = np.random.normal(0, beta*0.05, size=(N,N))
        np.fill_diagonal(pert, 0)
        ones += pert

    z = np.zeros((n_samples, N))
    
    for i in range(n_samples):
        x1 = np.random.normal(size=(1,N))
        x2 = np.random.normal(size=(1,N))
        
        z[i,:] = alpha * x1 + np.matmul(x2, ones)  
        
    return z
    
    
def oscillation(m):
    
    """
        Calculate the oscillation of a matrix (off diagonal):
            max(m) - min(m)
            
        Inputs:
            (2darray) m - Matrix
            
        Output:
            (float) osc - oscillation
    """
    
    # using a mask to ignore diagonal
    mask = np.ones(m.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # min and max
    max_value = m[mask].max()
    min_value = m[mask].min()
    
    return max_value - min_value

def gap(m):
    """
        Calculate the gap between the sigma squared (diagonal of the matrix) and the 
        maximum value in the off diagonal
        
        Input: 
            (2darray) m - Matrix
            
        Output:
            (int) gap - gap between diagonal mean and max of off diagonal
    """
    
    # using a mask to ignore diagonal
    mask = np.ones(m.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # max
    max_value = m[mask].max()
    return np.mean(np.diagonal(m)) - max_value

def a_min(m):
    """
        Calculating the minimum value in A for connected pairs 
        
    """
    # using a mask to ignore diagonal
    mask = np.ones(m.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    min_value = min(i for i in m[mask] if i > 0)
    return min_value

def noise_metrics(noise):
    """
        Input: 
            noise (2darray) - Noise matrix 
            
        Definitions: 
            C  : Covariance Matrix
            off(C): Off diagonal entries of C
            
        Metrics:
            Oscillation: max(off(Covariance)) - Min(off(cov))
    """
    C = np.cov(noise.T)
    
    osc = oscillation(C)
    sigma_gap = gap(C)
    
    return f"Oscillation: {osc}\nSigma Gap Squared: {sigma_gap}\nRatio: {osc/sigma_gap}\n"


def a_metrics(A, rho):
    """
    Input:
        A (2darray) - A Matrix
        rho (float) - float between 0 and 1 (its the sum of any row of A)
        
    Conditions:
        A = rho * Ã  where Ã : stochastic matrix
        0 < rho < 1
        
    Metric:
        ratio = (a_min / 2) *  ( (1 - rho^2) / (rho* (1 + rho^2) ) )
    """
    # calculate a min
    a_menor = a_min(A)
    
    # calculate the ratio
    ratio = (a_menor/2) * ((1-rho**2)/(rho*(1+rho**2)))
    
    
    return f"A Min: {a_menor}\nRho: {rho}\nRatio: {ratio}\n"

def get_acc_idgap(pred, y, dis_mpred, con_mpred):
    # - - - testing 
    #Divide the connected and disconnected pairs values
    test_size = len(y)
    idd = y < 2
    idd = np.squeeze(idd)

    idc = y > 1
    idc = np.squeeze(idc)

    nc = np.sum(idc)

    # predict and Normalize the values
    true = y - 1

    #Initialize the structures to save the data
    con = np.zeros((nc))
    dis = np.zeros((test_size-nc))

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

    #Compute metrics
    idgap = maxt - ((mint+maxt)/2)

    #Predicts the weight of the connection and classifies the data
    truec_nn = np.sum(dis_mpred<0.5)
    trued_nn = np.sum(con_mpred>0.5)

    tall = truec_nn + trued_nn

    acc = tall/(len(dis_mpred)+len(con_mpred))*100


    #If the id gap is negative then we cant correctly classify the pairs as disconnected or connected
    if(idgap < 0):
        idgap=0
    
    return acc, idgap

def load_pickles(name):
    pkl_file = open(name, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

def normalize_features(features, tsize):
    """
        Normalize the features based on the number of terms of correlate
    """
    offset = features.shape[1] // 2
    return features / np.arange(tsize-offset, tsize+offset)


def get_upper_tr(M):
    """
        Return upper triangle of a matrix
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

def get_off_diagonal(M):
    """
        Return the off diagonal entries of a matrix 
    """
    sz = M.shape[0]
    dsize = (sz*sz)-sz       #Number of elements excluding the diagonal
    vec = np.zeros((dsize,))
    counter = 0
    for i in range(sz):
        for j in range(sz):
            if (i != j):
                vec[counter,] = M[j,i]
                counter += 1
    return vec


def cluster_predictions(preds, method='gmm'):
    if method=='gmm':
        gmm = GaussianMixture(n_components=2, n_init=10)
        gmm.fit(preds)

        # norm
        centroids = gmm.means_
        if np.abs(centroids[0]) > np.abs(centroids[1]) :
            gmm.means_ = np.array([centroids[1],centroids[0]])

        cluster_preds = gmm.predict(preds) + 1
    else: 
        kmeans = KMeans(n_clusters=2, n_init='auto')
        kmeans.fit(preds)

        # norm
        centroids = kmeans.cluster_centers_
        if np.abs(centroids[0]) > np.abs(centroids[1]) :
            kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

        cluster_preds = kmeans.predict(preds) + 1
        
    return cluster_preds

# %% Metrics
from sklearn.metrics import confusion_matrix

def specificity_score(y_pred, y_true) -> float:
    """
        probability of a negative test result, 
        conditioned on the individual truly being 
        negative.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if tn+fp==0:
        return 0
    return tn / (tn+fp)

def sensitivity_score(y_pred, y_true) -> float:
    """
        probability of a positive test result, 
        conditioned on the individual truly being 
        positive.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if tp+fn==0:
        return 0
    return tp / (tp+fn)
    

#%% Get estimators prediction 
def get_r1_minus_r3_preds(timeseries, method='gmm'):
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
    preds = cluster_predictions(preds)
    
    return preds

def get_r1_preds(timeseries, method='gmm'):
    z=timeseries.T
    sz, tsize = z.shape

    # 1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)

    preds = get_upper_tr(R1).reshape((-1,1))
    preds = cluster_predictions(preds)
    
    return preds

def get_r0_inv_preds(timeseries, method = 'gmm'):
    z=timeseries.T
    tsize = z.shape[1]
    sz, tsize = z.shape

    identity = np.eye(sz, sz)

    # 0-lag correlation matrix
    R0 = np.matmul(z,z.T)/tsize
    try:
        r_inv = np.linalg.inv(R0)
    except Exception:
        r_inv = np.linalg.lstsq(R0,identity, rcond=None)[0]
    
    preds =  get_upper_tr(r_inv).reshape((-1,1))
    preds = cluster_predictions(preds)

    
    return preds


def get_granger_preds(timeseries, method = 'gmm'):
    z=timeseries.T
    sz, tsize = z.shape

    # 0-lag correlation matrix
    R0=np.matmul(z,z.T)/tsize

    # 1-lag correlation matrix
    z1=z[:,2:tsize]
    z2=z[:,1:tsize-1]
    R1=np.matmul(z1,z2.T)/(tsize-1)
    identity = np.eye(sz, sz)
    # R1*inv(R0)
    try:
        r_inv = np.linalg.inv(R0)
    except Exception:
        r_inv = np.linalg.lstsq(R0,identity, rcond=None)[0]
    r1_r0 =  np.matmul(R1,r_inv)

    preds = get_upper_tr(r1_r0).reshape((-1,1))
    preds = cluster_predictions(preds)

    return preds



def get_inverted_features(timeseries, n=200, undirected=True):
    '''
    Extracts the cross correlation and the inverse of the cross correlation lags

        Parameters:
                timeseries (1darray): Matrix with the observed node time series
                n (int): number of desired features
        Returns:
                features (2darray): Matrix containing a fecture vector for each pair of nodes
    '''
    z = timeseries.T  
    tsize = z.shape[1]
    sz = z.shape[0]
    identity = np.eye(sz, sz)

    if undirected==False:
        upper = int((sz*sz)-sz)
        func = get_off_diagonal
    else:
        upper = int(sz*(sz-1)/2)
        func = get_upper_tr

    features = np.zeros((upper,n))
    counter=0
    for offset in range(n//4):
        # positive lags
        z1 = z[:,offset+1:tsize]
        z2 = z[:,1:tsize-offset]
        R = np.matmul(z1,z2.T)/(tsize-offset)
        r = func(R)
        try:
            r_inv = func(np.linalg.inv(R))
        except Exception:
            r_inv = func(np.linalg.lstsq(R,identity, rcond=None)[0])
        features[:,counter] = r
        features[:,counter+1] = r_inv

        # negative lags
        z1 = z[:,offset+1:tsize]
        z2 = z[:,1:tsize-offset]
        R = np.matmul(z2,z1.T)/(tsize-offset)
        try:
            r_inv = func(np.linalg.inv(R))
        except Exception:
            r_inv = func(np.linalg.lstsq(R,identity, rcond=None)[0])
        r = func(R)
        features[:,counter+2] = r
        features[:,counter+3] = r_inv
        counter += 4
    return features


def extract_cross_correlation_unscaled_features(A,zz,n,n_features, undirected=True):

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

    if undirected==False:
        upper = int((n*n)-n)
    else:
        upper = int(n*(n-1)/2)

    data_nn = np.zeros((n_features,upper))
    target = np.zeros((1,upper))

    counter = 0
    offset = n_features//2

    #Go through each pair and compute the time laged cross-correlation
    if undirected:
        for j in range(n):
            for k in range(j+1,n):
                aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
                data_nn[:,counter] = aux[tsize-offset:tsize+offset]
                target[0,counter] = A[j,k]
                counter = counter + 1
    else:
        for j in range(n):
            for k in range(n):
                if j!=k:
                    aux = signal.correlate(zz[:,j],zz[:,k], mode="full")
                    data_nn[:,counter] = aux[tsize-offset:tsize+offset]
                    target[0,counter] = A[j,k]
                    counter = counter + 1

    data_nn=data_nn.T

    y=target>0
    y=y.astype(int)+1

    y=y.T
    return data_nn,y

def save_pickle(name,comparison_data):
    '''
    Saves the performance metrics of all estimators

        Parameters:
                comparison_data (2darray): Performance of different estimators over a range of number of samples

    '''
    output = open(name, 'wb')
    pickle.dump(comparison_data, output)
    output.close()
    

def get_model_architecture(n_features):
    from keras.layers import Dense, Dropout
    from keras.models import Sequential
    #CNN architecture
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(n_features,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(200, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# %% main
if __name__ == "__main__":
    print("Nothing to run")
