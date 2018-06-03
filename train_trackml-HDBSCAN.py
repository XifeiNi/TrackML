# The Mean Squares - TrackML Challenge 2018

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import hdbscan
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

# update path according to your folder setup
path_to_train = "/train_sample/train_100_events"
event_prefix = "event000001000"
hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

class Clusterer(object):

    def __init__(self,rz_scales=[0.65, 0.965, 1.528]):                        
        self.rz_scales=rz_scales
    
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            x = M[index]
            norms[i] = self._test_quadric(x)
        threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 6
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters==cluster]=0            
    def _test_quadric(self,x):
        Z = np.zeros((x.shape[0],10), np.float32)
        Z[:,0] = x[:,0]**2
        Z[:,1] = 2*x[:,0]*x[:,1]
        Z[:,2] = 2*x[:,0]*x[:,2]
        Z[:,3] = 2*x[:,0]
        Z[:,4] = x[:,1]**2
        Z[:,5] = 2*x[:,1]*x[:,2]
        Z[:,6] = 2*x[:,1]
        Z[:,7] = x[:,2]**2
        Z[:,8] = 2*x[:,2]
        Z[:,9] = 1
        v, s, t = np.linalg.svd(Z,full_matrices=False)        
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index,:]        
        norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
        return norm

    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
       
        return X
    
    def processLabels(self, ht):
    #helix transform equations    
        ht['rt'] = np.sqrt(ht.x**2+ht.y**2)
        ht['a0'] = np.arctan2(ht.y,ht.x)
        ht['r'] = np.sqrt(ht.x**2+ht.y**2+ht.z**2)
        ht['z1'] = ht.z/ht['rt'] 
        ht['z2'] = ht.z/ht['r']
        dz0 = -0.00070
        stepdz = 0.00001
        stepeps = 0.000005
        inv = 1
        for i in tqdm(range(100)):
            inv = inv * -1
            dz = inv*(dz0 + i*stepdz)
            ht['a1'] = ht['a0']+dz*ht['z']*np.sign(ht['z'].values)
            ht['sina1'] = np.sin(ht['a1'])
            ht['cosa1'] = np.cos(ht['a1'])
            ss = StandardScaler()
            dfs = ss.fit_transform(ht[['sina1','cosa1','z1','z2']].values)
            c = np.array([1.0,1.0,0.4,0.4])
            for j in range(np.shape(dfs)[1]):
                dfs[:,j] *= c[j]
            clusters = DBSCAN(eps=0.0035+i*stepeps,min_samples=1,metric='euclidean',n_jobs=8).fit(dfs).labels_
            if i==0:
                ht['s1']= clusters
                ht['N1'] = ht.groupby('s1')['s1'].transform('count')
            else:
                ht['s2'] = clusters
                ht['N2'] = ht.groupby('s2')['s2'].transform('count')
                max_s1 = ht['s1'].max()
                cond = np.where(((ht['N2'].values>ht['N1'].values) & (ht['N2'].values<20)))
                s1 = ht['s1'].values
                s1[cond] = ht['s2'].values[cond]+max_s1
                ht['s1'] = s1
                ht['s1'] = ht['s1'].astype('int64')
                self.clusters = ht['s1'].values
                ht['N1'] = ht.groupby('s1')['s1'].transform('count')
        return ht['s1'].values
    
    def predict(self, hits):         
        self.clusters = self.processLabels(hits)                
        X = self._preprocess(hits) 
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50)
        labels = np.unique(self.clusters)
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)
            self._eliminate_outliers(labels,X)
            max_len = np.max(self.clusters)
            self.clusters[self.clusters==0] = cl.fit_predict(X[self.clusters==0])+max_len
            labels = np.unique(self.clusters)
        return self.clusters
    
#model = Clusterer()
#labels = model.predict(hits)

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

#submission = create_one_event_submission(0, hits, labels)
#score = score_event(truth, submission)
#print("Your score: ", score)    

dataset_submissions = []
dataset_scores = []
for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
    # Track pattern recognition
    model = Clusterer()
    labels = model.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    dataset_submissions.append(one_submission)

    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)

    print("Score for event %d: %.8f" % (event_id, score))
print('Mean score: %.8f' % (np.mean(dataset_scores)))

path_to_test = "/test/test"
test_dataset_submissions = []


for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

    # Track pattern recognition 
    model = Clusterer()
    labels = model.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    test_dataset_submissions.append(one_submission)
    print('Event ID: ', event_id)

# Create submission file
submission = pd.concat(test_dataset_submissions, axis=0)
submission.to_csv('submission.csv', index=False)