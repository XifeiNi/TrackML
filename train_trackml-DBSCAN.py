# The Mean Squares - TrackML Challenge 2018
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

# update path according to your folder setup
path_to_train = "/train_sample/train_100_events"
event_prefix = "event000001000"
hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

class Clusterer(object):

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
            sc = np.array([1.0,1.0,0.4,0.4])
            for j in range(np.shape(dfs)[1]):
                dfs[:,j] *= sc[j]
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
        return self.clusters

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission  

# Training/Validation set
train_submissions = []
dataset_scores = []
for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
    # Track pattern recognition
    model = Clusterer()
    labels = model.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    train_submissions.append(one_submission)

    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)

    print("Score for event %d: %.8f" % (event_id, score))
print('Mean score: %.8f' % (np.mean(dataset_scores)))

# Test set
path_to_test = "/test/test"
test_submission = []

# 125 test events
for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

    # Track pattern recognition 
    model = Clusterer()
    labels = model.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    test_submission.append(one_submission)
    print('Event ID: ', event_id)

# Create submission file
submission = pd.concat(test_submission, axis=0)
submission.to_csv('submission-001.csv', index=False)    #Size about 200MB