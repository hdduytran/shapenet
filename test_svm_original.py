import wrappers_original
import json
from uea_original import load_UEA_dataset
import os
import sys


ratio = float(sys.argv[2])
dataset = sys.argv[1]

train, train_labels, test, test_labels = load_UEA_dataset(
        '/content/datasets', dataset, ratio
    )
clf = wrappers_original.CausalCNNEncoderClassifier()
hf = open(
    os.path.join(
        '/content/shapenet/OriginalShapeNet/models/model_10', dataset + '_parameters.json'
    ), 'r'
)
hp_dict = json.load(hf)
hf.close()
hp_dict['cuda'] = True
hp_dict['gpu'] = 0
clf.set_params(**hp_dict)
clf.load(os.path.join('/content/shapenet/OriginalShapeNet/models/model_10', dataset))

for cluster_num, final_shapelet in [(100, 50)]:
    # shapelet discovery
    shapelet, shapelet_dim, utility_sort_index = clf.shapelet_discovery(train, train_labels, cluster_num=cluster_num, batch_size=50)

    # # # shapelet transformation
    features = clf.shapelet_transformation(train, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num=final_shapelet)

    # # # SVM classifier training
    clf.classifier = clf.fit_svm_linear(features, train_labels)
    print(f"svm linear Accuracy {cluster_num} - {final_shapelet}: "+str(clf.score(test, test_labels, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num=final_shapelet)))