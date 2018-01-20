import csv
import numpy as np

def vectorizeFeatures(path, res):
    ftr_list = []

    f = open(path)
    reader = csv.reader(f)
    for row in reader:
        row_np = np.array(row)
        ftr_list.append(row_np)
    f.close()

    ftr_np = np.array(ftr_list)
    print ftr_np
    np.save(res, ftr_np)

def vectorizeLabels(path, res):
    lbl_list = []

    f = open(path)
    reader = csv.reader(f)
    for row in reader:
        lbl = row[1].split('/')[2]
        lbl_list.append(lbl)
    f.close()

    lbl_np = np.array(lbl_list)
    print lbl_np
    np.save(res, lbl_np)

vectorizeLabels('generated-embeddings/labels.csv', 'label.npy')
vectorizeFeatures('generated-embeddings/reps.csv', 'feature.npy')
