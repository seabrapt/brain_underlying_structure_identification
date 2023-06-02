"""
    python file that contains all general functions like:
        - Generating a graph
        - Generating A matrix
        - Generating Time-series
        - Creating dataset

        # TODO: Check functions
"""

# Imports
import numpy as np
from scipy import signal
import pickle


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




def tsg(A,tsize,x0,noise):
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
        for j in range(sz):
            x[i,j] = np.dot(A[j,:],x[i-1,:]) + noise[i,j]

    return x


def tsg2(A,tsize,x0,noise):
    '''
    UPDATED TSG - FASTER
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



def create_dataset(sz,p,c,rho,tsize,x0,undirected, noise):
    '''
    Generates the synthectic data, extracts the features and returns the tranning/testing dataset

        Parameters:
                sz (int): Number of nodes
                p (int): Probability of existing an edge between each pair of nodes
                c,rho (int): Numbers between 0 and 1, to make the spectral radius < 1
                tsize (int): Time series size - number of samples
                x0 (int): Initial condition x(0), in this case is zero
                qsi (int): Noise standart deviation

        Returns:
                data (2darray): Matrix containing the feature-vectors between each pair of nodes
                target (1darray): Ground-truth - pairs are connected or disconnected
    '''
    #Generate the adjacency and A matrices
    adj = get_adjacency(sz,p,undirected)
    A = get_A(adj,c,rho)

    #Is the graph undirected or directed
    if(undirected):

        #Create data structures
        upper = int(sz*(sz-1)/2)  #Number of elements in the upper matrix
        data = np.zeros((200,upper))
        target = np.zeros((1,upper))

        #Generates the synthetic time series
        x = tsg2(A,tsize,x0,noise)

        #Goes through each pair (of the upper matrix) and computes the time laged cross-correlation (excludes diagonal)
        counter = 0
        for j in range(sz):
            for k in range(j+1,sz):
                #Compute the cross correlation
                aux = signal.correlate(x[:,j],x[:,k], mode="full")

                #Extracts the first negative and positive lags
                data[:,counter] = aux[tsize-100:tsize+100]

                #Saves the data
                target[0,counter] = A[j,k]
                counter = counter + 1
    else:
        #Create data structures
        dsize = (sz*sz)-sz       #Number of elements excluding the diagonal
        data = np.zeros((200,dsize))
        target = np.zeros((1,dsize))

        #Generates the synthetic time series
        x = tsg(A,tsize,x0,qsi,noise)

        #Goes through each pair and computes the time laged cross-correlation (excludes diagonal)
        counter = 0
        for j in range(sz):
            for k in range(sz):
                if(j!=k):
                    #Computes the cross correlation
                    aux = signal.correlate(x[:,j],x[:,k], mode="full")

                    #Extracts the firs negative and positive lags
                    data[:,counter] = aux[tsize-100:tsize+100]

                    #Saves the data
                    target[0,counter] = A[j,k]
                    counter = counter + 1
    return data,target, A



def get_time_series(sz,p,c,rho,tsize,x0,qsi,undirected):
    adj = get_adjacency(sz,p,undirected)
    A = get_A(adj,c,rho)
    x = tsg(A,tsize,x0,noise)
    return x



# - - - - - NOISE FUNCTIONS - - - - - - 
# np.set_printoptions(precision=5, suppress=True)

def generate_noise(N, n_samples, alpha, beta):
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

if __name__ == "__main__":
    print("Nothing to run")
