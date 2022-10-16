import numpy as np

def get_top_k(representation_all, k = 10):
    top_k_feature_maps = []
    num_cluster = representation_all.shape[0]
    for i in range(num_cluster):
        cluster_i = representation_all[i] #[175,10,fea_size]
        centroid_i = np.mean(cluster_i, axis=0).reshape(1,-1)
        dist_i = []
        for feature_map in cluster_i:
            dist_i.append(np.linalg.norm(feature_map - centroid_i))
        arg_sorted_dist_i = np.argsort(dist_i)
        top_k_feature_maps.append([np.array(cluster_i[arg_sorted_dist_i][:k])])
    return np.array(top_k_feature_maps)

def dis1(query_topk, proto):
    '''Cách 1: prototype = Trung bình top K shapelet, query chọn top k shapelet, tính trung bình. Tính khoảng cách từ TB của prototype đến TB của query -> ra một số	'''
    mean_query = np.mean(query_topk,axis = 2)
    mean_query = np.repeat(mean_query,3,axis = 1)
    mean_proto = np.mean(proto, axis = 2)
    # print('mean proto:', mean_proto.shape)
    # print('mean query:', mean_query.shape)

    dis = np.power(mean_query-mean_proto,2).sum(-1)
    # print('dis',dis.shape,dis)
    nearest = np.argmin(dis,axis = 1)
    # print('nearest:', nearest)
    prediction = nearest
    return prediction

def dis2(query_topk,proto):
    mean_query = np.mean(query_topk,axis = 2,keepdims=True)
    # print('mean query:', mean_query.shape)
    mean_query = np.repeat(mean_query,3,axis = 1)
    # print('mean query:', mean_query.shape)
    mean_query = np.repeat(mean_query,query_topk.shape[2],axis = 2)
    # print('mean query:', mean_query.shape)
    dis = np.power((proto - mean_query), 2).sum(-1)
    # print('dis', dis.shape)
    nearest = np.argmin(dis,axis = 1)
    # #[]
    # print('nearest:', nearest)
    # print(np.unique(nearest,return_counts= True))

    prediction = []
    for i in nearest:
        # print(np.unique(i,return_counts=True))
        classes, count_cls = np.unique(i,return_counts=True)
        # print('cls', classes,count_cls)
        pred_cls = classes[np.argmax(count_cls)]
        # print(pred_cls)
        prediction.append(pred_cls)
    prediction = np.array(prediction)
    return prediction
    
def dis3(query_topk,proto):
    mean_query = np.mean(query_topk,axis = 2,keepdims=True)
    mean_query = np.repeat(mean_query,3,axis = 1)
    mean_query = np.repeat(mean_query,query_topk.shape[2],axis = 2)
    # print('mean query:', mean_query.shape)
    dis = np.power((proto - mean_query), 2).sum(-1)
    dis = np.min(dis, axis = 2)
    # print('dis', dis.shape)
    nearest = np.argmin(dis,axis = 1)
    # print('nearest', nearest)
    return nearest 

from sklearn.neighbors import KNeighborsClassifier

def dis4(query_topk,proto,train_labels,n=3):
    knn = KNeighborsClassifier(n_neighbors=n)

    num_classes = len(np.unique(train_labels))

    proto = np.reshape(proto,(num_classes*query_topk.shape[2],query_topk.shape[-1]))
    proto_labels = np.array(np.unique(train_labels).repeat(query_topk.shape[2]))
    # print('protoo',proto.shape)
    # print(proto_labels)

    mean_query = query_topk.squeeze().mean(axis=1)
    # print(mean_query.shape)

    knn.fit(proto,proto_labels)

    prediction = knn.predict(mean_query)
    # print(prediction)
    return prediction

def dis5(query_topk,proto,train_labels,n=3):
    knn = KNeighborsClassifier(n_neighbors=n)

    num_classes = len(np.unique(train_labels))

    proto = np.reshape(proto,(num_classes*query_topk.shape[2],query_topk.shape[-1]))
    proto_labels = np.array(np.unique(train_labels).repeat(query_topk.shape[2]))
    # print('protoo',proto.shape)
    # print(proto_labels)

    mean_query = query_topk.squeeze()
    print(mean_query.shape)

    knn.fit(proto,proto_labels)

    prediction = []
    for i in mean_query:
        pred = knn.predict(i)
        classes, count_cls = np.unique(pred,return_counts=True)
        # print('cls', classes,count_cls)
        pred_cls = classes[np.argmax(count_cls)]
        prediction.append(pred_cls)
    # print(prediction)
    return prediction