import numpy as np
from sklearn.metrics import accuracy_score
import json
import sys

from distances_mst import *

import wrappers_original
from uea_original import load_UEA_dataset

model_path = sys.argv[1]
ratio = float(sys.argv[3])
dataset = sys.argv[2]

model = wrappers_original.CausalCNNEncoderClassifier(out_channels=5)

hf = open(
    model_path + f'/{dataset}_parameters.json'
)
hp_dict = json.load(hf)
hf.close()
model.set_params(**hp_dict)
model.load(model_path + f'/{dataset}')

train, train_labels, test, test_labels = load_UEA_dataset(
        '/content/datasets', f'{dataset}', ratio
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

# query_embs = model.inference(query) # (175,251)
shapelet, shapelet_dim, utility_sort_index = model.shapelet_discovery(train, train_labels, cluster_num=100, batch_size=50)
# print(query_embs.shape,type(query_embs))
for k in [50]:
    print('k: ',k)
    train_features = model.shapelet_transformation(train, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num=k)
    test_features = model.shapelet_transformation(test, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num=k)
    # query_topk = get_top_k(query_embs,k=k)
    # print('query_topk',query_topk.shape)

    # proto = model.shapelet_discovery(train,train_labels,num_class,k=k)
    # proto = np.array([x[1] for x in proto])
    # proto = np.expand_dims(proto, axis = 0)
    # print('proto:',proto.shape,type(proto))
    
    # prediction = dis1(train_features, test_features)
    
    # accuracy_dis1 = accuracy_score(labels,prediction)
    # # print(prediction)
    # print('accuracy_dis1',accuracy_dis1)

    # prediction = dis2(train_features, test_features)
    # accuracy_dis2 = accuracy_score(labels,prediction)
    # # print(prediction)
    # print('accuracy_dis2',accuracy_dis2)

    # prediction = dis3(train_features, test_features)
    # accuracy_dis3 = accuracy_score(labels,prediction)
    # # print(prediction)
    # print('accuracy_dis3',accuracy_dis3)
    dis4s = []
    dis5s = []
    for n in [1,3,5,7,10,15]:
        if n >k:
            break
        # print('\tn=',n)
        prediction = dis4(test_features, train_features, train_labels,n=n)
        accuracy_dis4 = accuracy_score(labels,prediction)
        # print(prediction)
        # print('accuracy_dis4',accuracy_dis4)
        dis4s.append(accuracy_dis4)
        prediction = dis5(test_features, train_features, train_labels,n=n)
        accuracy_dis5 = accuracy_score(labels,prediction)
        # print(prediction)
        # print('accuracy_dis5',accuracy_dis5)
        dis5s.append(accuracy_dis5)
    print('accuracy_dis4 ', dis4s)
    print('accuracy_dis5 ', dis5s)