#Run convert to numpy before running
#This code performs image resize on given numpy array
#Then PCA is performed on the array and best number of components are 
#identified using a test test of just 1000 examples Accuray is reported as score
import numpy as np
from sklearn import linear_model
import datetime
import scipy
from sklearn.decomposition import PCA

# iterates over the range of component values
for i in range(5):
    X=np.load('train_inputs.npy') 
    T=np.load('train_outputs.npy')
    X=scipy.misc.imresize(X, (10001,576), interp='bilinear', mode=None)
    pca = PCA(n_components=150+50*i) # specify the number of components required
    pca.fit(X)
    X=pca.transform(X)
    #print (pca.explained_variance_ratio_)
    X_trn=X[0:9900,:]
    T_trn=T[0:9900]
    X_te=X[9901:,:]
    T_te=T[9901:]
    print X_trn.shape
    logistic=linear_model.LogisticRegression()
    print logistic.fit(X_trn,T_trn).score(X_te,T_te)
print "Done"
