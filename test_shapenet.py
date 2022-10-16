import numpy as np
from sklearn.metrics import accuracy_score
import json
import sys

from distances import *

import wrappers
from uea import load_UEA_dataset

model_path = sys.argv[1]
model = wrappers.CausalCNNEncoderClassifier(out_channels=5)

hf = open(
    model_path + '/ArrowHead_parameters.json'
)
hp_dict = json.load(hf)
hf.close()
model.set_params(**hp_dict)
model.load(model_path + '/ArrowHead')

train, train_labels, test, test_labels = load_UEA_dataset(
        'data', 'ArrowHead'
    )
num_class = len(np.unique(test_labels))
num_query = test.shape[0]
# num_query = 2
# print(test_labels)
query = test[:num_query]
labels = test_labels[:num_query]
# print('querry:', query.shape)
# print('labels: ', labels)

f = open(model_path + '/results.txt','w')

query_embs = model.inference(query) # (175,251)
# print(query_embs.shape,type(query_embs))
for k in [2,5,10,15,20]:
    print('k: ',k,file=f)
    query_topk = get_top_k(query_embs,k=k)
    # print('query_topk',query_topk.shape)

    proto = model.shapelet_discovery(train,train_labels,num_class,k=k)
    proto = np.array([x[1] for x in proto])
    proto = np.expand_dims(proto, axis = 0)
    # print('proto:',proto.shape,type(proto))

    prediction = dis1(query_topk,proto)
    accuracy_dis1 = accuracy_score(labels,prediction)
    # print(prediction)
    print('accuracy_dis1',accuracy_dis1,file=f)

    prediction = dis2(query_topk,proto)
    accuracy_dis2 = accuracy_score(labels,prediction)
    # print(prediction)
    print('accuracy_dis2',accuracy_dis2,file=f)

    prediction = dis3(query_topk,proto)
    accuracy_dis3 = accuracy_score(labels,prediction)
    # print(prediction)
    print('accuracy_dis3',accuracy_dis3,file=f)
    dis4s = []
    dis5s = []
    for n in [1,3,5,7,10,15]:
        if n >k:
            break
        # print('\tn=',n,file=f)
        prediction = dis4(query_topk,proto,train_labels,n=n)
        accuracy_dis4 = accuracy_score(labels,prediction)
        # print(prediction)
        # print('accuracy_dis4',accuracy_dis4)
        dis4s.append(accuracy_dis4)
        prediction = dis5(query_topk,proto,train_labels,n=n)
        accuracy_dis5 = accuracy_score(labels,prediction)
        # print(prediction)
        # print('accuracy_dis5',accuracy_dis5)
        dis5s.append(accuracy_dis5)
    print('accuracy_dis4 ', dis4s,file=f)
    print('accuracy_dis5 ', dis5s,file=f)