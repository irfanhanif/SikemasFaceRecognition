import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ftr = np.load('feature.npy')
lbl = np.load('label.npy')

clf = RandomForestClassifier(n_estimators=10000, oob_score=True).fit(ftr, lbl)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
