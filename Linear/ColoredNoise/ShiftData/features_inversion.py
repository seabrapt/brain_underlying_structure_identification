# %% Imports
import numpy as np
from keras.models import *

from joblib import dump, load

import sys
sys.path.insert(1, '../')

from functions import *
from estimators import *
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from tqdm import trange
import gc

# %% models 
model_betas = load_model('model_ffnn_50p_200f_ss_ss_super')
model_beta5_ss = load_model('models7/model_ffnn_50p_5b_200f_ss')
model_beta5_ss_ss = load_model('models7/model_ffnn_50p_5b_200f_ss_ss')
model_ss = load_model('models7/model_ffnn_50p_200f_ss')
model_ss_ss = load_model('models7/model_ffnn_50p_200f_ss_ss')

model_beta1_ss = load_model('models7/model_ffnn_50p_1b_low_samples_200f_ss')
model_beta1_ss_ss = load_model('models7/model_ffnn_50p_1b_low_samples_200f_ss_ss')

model_inv_ss = load_model('models8/model_ffnn_50p_1b_200f_inv_ss')
model_inv_ss_ss = load_model('models8/model_ffnn_50p_1b_200f_inv_ss_ss')

cnn = load_model('models/model_50p_200f_max_alpha1')

# %% parameters
min_size = 5000
tsize = 500000
max_size= tsize
step = 5000
tssize = np.arange(min_size,max_size,step=step) 
print(tssize)

#Parameters
sz = 100
ss = 0
se = 10
c = 0.9
rho = 0.75

#Define the range of noise variance
qsi = 1
x0 = 0            #Initial condition

#True if the graph is undirected, False if not
undirected = True

# variables
betas = [0, 1, 5, 10]
p = 0.7

# %% run
from sklearn.metrics import accuracy_score

nruns = 10
comparison_data = np.zeros((len(tssize), nruns, len(betas), 12, 1))

 #Make several runs
for i in (t :=  trange(nruns)):
    # generate an A matrix
    adj = get_adjacency(sz, p, undirected)
    A = get_A(adj,c,rho)

    for beta in range(len(betas)):
        # generate noise
        noise = generate_noise(sz, tsize, qsi, betas[beta]) 
        
        # generate the time series
        data2 = tsg2(A,tsize,x0,noise)

        for j in range(len(tssize)):
            data_aux=np.copy(data2)
            data=data_aux[0:(tssize[j]),ss:se]
            As = A[ss:se,ss:se]

            # features ss
            features_unscaled, y = extract_cross_correlation_unscaled_features(As,data,se-ss, 200)
            features_scaled = normalize_features(features_unscaled, tssize[j])
            scaler = StandardScaler()
            features_ss = scaler.fit_transform(features_scaled)

            # features max
            features_max = features_unscaled / np.max(features_unscaled)
            features_max = features_max.reshape(features_max.shape[0],features_max.shape[1],1)

            # features ss ss
            scaler_ts = StandardScaler()
            x_scaled = scaler_ts.fit_transform(data)
            features_unscaled, y = extract_cross_correlation_unscaled_features(As,x_scaled,se-ss, 200)
            features_scaled = normalize_features(features_unscaled, tssize[j])
            scaler = StandardScaler()
            features_ss_ss = scaler.fit_transform(features_scaled)

            # - - - FFNN SS
            preds = model_ss.predict(features_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 0, 0] = acc


            # - - - FFNN SS SS
            preds = model_ss_ss.predict(features_ss_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 1, 0] = acc


            # - - - FFNN SS SS various betas
            preds = model_betas.predict(features_ss_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 2, 0] = acc


            # - - - FFNN SS BETA = 5
            preds = model_beta5_ss.predict(features_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 3, 0] = acc


            # - - - FFNN SS SS BETA = 5
            preds = model_beta5_ss_ss.predict(features_ss_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 4, 0] = acc


            # - - - FFNN SS BETA = 1
            preds = model_beta1_ss.predict(features_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 5, 0] = acc


            # - - - FFNN SS SS BETA = 1
            preds = model_beta1_ss_ss.predict(features_ss_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 6, 0] = acc


            # - - - FFNN SS INVERTED
            features_inverted = get_inverted_features(data,100)
            scaler = StandardScaler()
            features_ss = scaler.fit_transform(features_inverted)
            preds = model_inv_ss.predict(features_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 7, 0] = acc


            # - - - FFNN SS SS INVERTED
            features_inverted = get_inverted_features(x_scaled,100)
            scaler = StandardScaler()
            features_ss = scaler.fit_transform(features_inverted)
            preds = model_inv_ss_ss.predict(features_ss, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 8, 0] = acc

            # - - - granger - kmeans
            preds = get_granger_preds(data, method='kmeans')
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 9, 0] = acc

            # - - - granger - gmm
            preds = get_granger_preds(data, method='gmm')
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 10, 0] = acc

            # - - - CNN Max
            preds = cnn.predict(features_max, verbose=0)
            kmeans = KMeans(n_clusters=2, n_init='auto')
            kmeans.fit(preds)

            centroids = kmeans.cluster_centers_
            if centroids[0] > centroids[1] :
                kmeans.cluster_centers_ = np.array([centroids[1],centroids[0]])

            preds = kmeans.predict(preds) + 1
            acc = accuracy_score(y,preds)
            comparison_data[j, i, beta, 11, 0] = acc



    save_name = f"feature_inversion_70p_3_{i}"
    save_pickle(save_name, comparison_data[:,0:i+1,:,:,:])
