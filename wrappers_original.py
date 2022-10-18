from cmath import inf
import math
import numpy as np
import torch
import random
import pandas as pd
import sklearn
import sklearn.linear_model
import joblib
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import timeit

import utils
import losses
import networks
import slide_original

class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length,
                 batch_size, epochs, lr,
                 encoder, params, in_channels, cuda=False, gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.loss = losses.triplet_original.PNTripletLoss(
            compared_length
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)

    def save_shapelet(self, prefix_file, shapelet, shapelet_dim):
        '''
        write the shapelet and its dimension to file
        '''
        # save shapelet
        fo_shapelet =open(prefix_file+"shapelet.txt", "w")
        for j in range(len(shapelet)):
            shapelet_tmp = np.asarray(shapelet[j])
            s = shapelet_tmp.reshape(1,-1)
            np.savetxt(fo_shapelet, s)

        fo_shapelet.close()

        # save shapelet variable
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "w")
        np.savetxt(fo_shapelet_dim, shapelet_dim)
        fo_shapelet_dim.close()

    def load_shapelet(self, prefix_file):
        '''
        load the shapelet and its dimension from disk
        '''
        # save shapelet
        fo_shapelet = prefix_file+"shapelet.txt"
        with open(fo_shapelet, "r") as fo_shapelet:
            shapelet = []
            for line in fo_shapelet:
                shapelet.append(line)
        fo_shapelet.close()

        # save shapelet dimension
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "r")
        shapelet_dim = np.loadtxt(fo_shapelet_dim)
        fo_shapelet_dim.close()

        return shapelet, shapelet_dim

    def save_encoder(self, prefix_file, ratio):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + f'_encoder_original_ratio_{ratio}.pth'
        )

    def save(self, prefix_file, ratio):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file, ratio)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_svm_linear(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an svm linear
        classifier.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        self.classifier = SVC(kernel='linear' ,gamma='auto')
        self.classifier.fit(features, y)

        return self.classifier

    def fit_encoder(self, ratio, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        n_folds = math.floor(X.shape[0] / math.ceil(X.shape[0] * 0.2))
        kfold = KFold(n_splits=n_folds, shuffle=True)
        train_torch_dataset = utils.Dataset(X)

        # train_torch_dataset = utils.Dataset(X)
        # train_generator = torch.utils.data.DataLoader(
        #     train_torch_dataset, batch_size=self.batch_size, shuffle=True
        # )
        # valid_torch_dataset = utils.Dataset(X_valid)
        # valid_generator = torch.utils.data.DataLoader(
        #     valid_torch_dataset, batch_size=self.batch_size, shuffle=True
        # )

        train_lossList = []
        valid_lossList = []
        #epochs = 0 # Number of performed epochs
        last_loss = inf # Early stopping
        patience = 3 # Early stopping
        # Encoder training
        for i in range(self.epochs):
            epoch_start = timeit.default_timer()
            batch_index = 1

            train_loss_per_fold = []
            valid_loss_per_fold = []
            for fold, (train_ids, test_ids) in enumerate(kfold.split(train_torch_dataset)):
                # Sample elements randomly from a given list of ids, no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
                # Define data loaders for training and testing data in this fold
                train_generator = torch.utils.data.DataLoader(
                                train_torch_dataset, 
                                batch_size=self.batch_size, sampler=train_subsampler)
                valid_generator = torch.utils.data.DataLoader(
                                train_torch_dataset,
                                batch_size=self.batch_size, sampler=test_subsampler)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(train_generator), epochs=self.epochs-i)
                train_loss_per_batch = []
                for batch in train_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    self.optimizer.zero_grad()
                    loss, dist_positive_list, dist_negative_list, dist_intra_positive_list, dist_intra_negative_list = self.loss(
                    batch, self.encoder, self.params, save_memory=save_memory
                    )
                    loss.backward()
                    self.optimizer.step()
                    scheduler.step()
                    train_loss_per_batch.append(loss.item())

                    if i < 10 or i%100 == 99:
                        print(f'epoch {i+1} - batch {batch_index}:')
                        # print(f'\tdist_positive: {dist_positive_list}\t|\tdist_negative: {dist_negative_list}')
                        # print(f'\tdist_intra_positive_list: {dist_intra_positive_list}\t|\tdist_intra_negative_list: {dist_intra_negative_list}')
                    batch_index += 1
                train_loss_per_fold.append(np.mean(train_loss_per_batch))

                epoch_end = timeit.default_timer()
                if i % 100 == 99:
                    print(f"epoch {i+1} time: {(epoch_end- epoch_start)/60}")
                if i == 399:
                    print('save')
                    torch.save(self.encoder.state_dict(), "ArrowHead" + '_' + self.architecture + '_encoder_400.pth')
                #epochs += 1
                
                # Early stopping
                valid_loss_per_batch = []
                for batch in valid_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    self.optimizer.zero_grad()
                    valid_loss, _, _, _, _ = self.loss(
                    batch, self.encoder, self.params, save_memory=save_memory
                    )
                    valid_loss_per_batch.append(valid_loss.item())
                valid_loss_per_fold.append(np.mean(valid_loss_per_batch))

            train_lossList.append(np.mean(train_loss_per_fold))
            valid_lossList.append(np.mean(valid_loss_per_fold))
            current_loss = np.mean(valid_loss_per_fold)

        
            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    loss_per_epoch = {'epochs': [j for j in range(1, i+2)], 'loss':train_lossList, 'valid_loss': valid_lossList}
                    loss_df = pd.DataFrame.from_dict(loss_per_epoch).to_csv(f'./shapenet_result/loss_per_epoch_newloss_01_ratio_{ratio}.csv', index=False)
                    return self.encoder
            else:
                trigger_times = 0
            last_loss = current_loss
            
        loss_per_epoch = {'epochs': [j for j in range(1, i+2)], 'loss':train_lossList, 'valid_loss': valid_lossList}
        loss_df = pd.DataFrame.from_dict(loss_per_epoch).to_csv(f'./shapenet_result/loss_per_epoch_newlossver2_ratio_{ratio}.csv', index=False)

        return self.encoder

    def fit(self, ratio, ind, X, y, test, test_labels, prefix_file, cluster_num, save_memory=False, verbose=False):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param test testing set.
        @param test_labels testing labels.
        @param prefix_file prefix path.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        final_shapelet_num = 50
        # Fitting encoder
        encoder_start = timeit.default_timer()
        self.encoder = self.fit_encoder(
                                        ratio, X, y=y, save_memory=save_memory, verbose=verbose
                                        )
        encoder_end = timeit.default_timer()
        print("encode time: ", (encoder_end- encoder_start)/60)

        # # shapelet discovery
        discovery_start = timeit.default_timer()
        shapelet, shapelet_dim, utility_sort_index = self.shapelet_discovery(X, y, cluster_num, batch_size=50)
        discovery_end = timeit.default_timer()
        print("discovery time: ", (discovery_end- discovery_start)/60)

        # # shapelet transformation
        transformation_start = timeit.default_timer()
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        transformation_end = timeit.default_timer()
        print("transformation time: ", (transformation_end - transformation_start)/60)

        # # SVM classifier training
        classification_start = timeit.default_timer()
        self.classifier = self.fit_svm_linear(features, y)
        classification_end = timeit.default_timer()
        print("classification time: ", (classification_end - classification_start)/60)
        score = str(self.score(test, test_labels, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num))
        print("svm linear Accuracy: "+score)
        fo_shapelet =open(prefix_file, "a")
        strings_res = f'ratio:\t {ratio}, ind:\t {ind}, score:\t {score}'
        fo_shapelet.write(strings_res)
        fo_shapelet.close()
        self.save_shapelet(prefix_file, shapelet, shapelet_dim)

        return self

    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        features = np.zeros((np.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                    count * batch_size: (count + 1) * batch_size
                ] = self.encoder(batch)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def shapelet_discovery(self, X, train_labels, cluster_num, batch_size = 50):
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet
        '''

        slide_num = 3
        alpha = 0.6
        count = 0
        X_slide_num = []
        gama = 0.5

        for m in range(slide_num):
            # slide the raw time series and the corresponding class and variate label
            X_slide, candidates_dim, candidates_class_label = slide_original.slide_MTS_dim_step(X, train_labels, alpha)
            # X_slide shape: (số lượng subsequence, chiều dài subsequence)
            # candidates_dim shape: (số lượng subsequence,)
            # candidates_class_label shape: (số lượng subsequence,)
            X_slide_num.append(np.shape(X_slide)[0])
            alpha = round(alpha - 0.2, 1)

            test = utils.Dataset(X_slide)
            test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)

            self.encoder = self.encoder.eval()

            # encode slide TS
            with torch.no_grad():
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # 2D to 3D
                    batch.unsqueeze_(1)
                    batch = self.encoder(batch)

                    if count == 0:
                        representation = batch.cpu().numpy()
                    else:
                        representation = np.concatenate((representation, batch.cpu().numpy()), axis=0)
                    count += 1
            
            count = 0
            # concatenate the new representation from different slides
            if m == 0 :
                representation_all = representation # (số lượng rep, chiều dài rep)
                representation_dim = candidates_dim # (số lượng rep, )
                representation_class_label = candidates_class_label # (số lượng rep, )
            else:
                representation_all = np.concatenate((representation_all, representation), axis = 0)
                representation_dim = representation_dim + candidates_dim
                representation_class_label = np.concatenate((representation_class_label, candidates_class_label), axis=0)

        # cluster all the new representations
        num_cluster = cluster_num
        kmeans = KMeans(n_clusters = num_cluster)
        kmeans.fit(representation_all)

        # init candidate as list
        candidate = []
        candidate_dim = np.zeros(num_cluster) # mỗi cluster chọn 1 candidate
        # two parts of utility function
        candidate_cluster_size = []
        candidate_first_representation = []
        utility = []

        # select the nearest to the centroid
        for i in range(num_cluster):
            candidate_cluster_size.append(representation_all[kmeans.labels_==i][:,0].size)
            dim_in_cluster_i = list()
            class_label_cluster_i = list()
            dist = math.inf
            for j in range(representation_all[kmeans.labels_==i][:,0].size): # Duyệt qua các representation có cluster label là i
                match_full = np.where(representation_all == (representation_all[kmeans.labels_==i][j]))
                match = np.unique(match_full)
                dist_tmp = np.linalg.norm(representation_all[kmeans.labels_==i][j] - kmeans.cluster_centers_[i])
                for k in range(match.shape[0]): 
                    dim_in_cluster_i.append(representation_dim[match[k]])
                    class_label_cluster_i.append(representation_class_label[match[k]])
                if dist_tmp < dist:
                    dist = dist_tmp

                    # record the first representation
                    tmp_candidate_first_representation = representation_all[kmeans.labels_==i][j]
                    # trace back the original candidates
                    nearest = np.where(representation_all == (representation_all[kmeans.labels_==i][j]))
                    sum_X_slide_num = 0
                    for k in range(slide_num):
                        sum_X_slide_num += X_slide_num[k]
                        if (nearest[0][0] < sum_X_slide_num):
                            index_slide = nearest[0][0] - sum_X_slide_num + X_slide_num[k]
                            X_slide_disc = slide_original.slide_MTS_dim(X, (0.6-k*0.2))
                            candidate_tmp = X_slide_disc[index_slide]
                            candidate_dim[i] = index_slide % np.shape(X)[1]
                            break
            class_label_top = (Counter(class_label_cluster_i).most_common(1)[0][1] / len(class_label_cluster_i))
            dim_label_top = (Counter(dim_in_cluster_i).most_common(1)[0][1] / len(dim_in_cluster_i))
            if (class_label_top < (1/np.unique(train_labels).shape[0])) or (dim_label_top < (1/np.shape(X)[1])) :
                del candidate_dim[-1]
                continue
            # append the first representation
            candidate_first_representation.append(tmp_candidate_first_representation)
            # list append method
            candidate.append(candidate_tmp)

        # utility
        for i in range(num_cluster):
            ed_dist_sum = 0
            for j in range(len(candidate_first_representation)):
                ed_dist_sum += np.linalg.norm(candidate_first_representation[i] - candidate_first_representation[j])
            utility.append(gama * candidate_cluster_size[i] + (1-gama) * ed_dist_sum)

        # sort utility namely candidate
        utility_sort_index = np.argsort(-np.array(utility))

        return candidate, candidate_dim, utility_sort_index

    def shapelet_transformation(self, X, candidate, candidate_dim, utility_sort_index, final_shapelet_num):
        '''
        transform the original multivariate time series into the new one vector data space
        transformed date label the same with original label
        '''
        # init transformed data with list
        feature = []

        # transform original time series
        for i in range(np.shape(X)[0]):
            for j in range(final_shapelet_num):
            #for j in range(len(candidate)):
                dist = math.inf
                candidate_tmp = np.asarray(candidate[utility_sort_index[j]])
                for k in range(np.shape(X)[2]-np.shape(candidate_tmp)[0]+1):
                    difference = X[i, int(candidate_dim[utility_sort_index[j]]), 0+k : int(np.shape(candidate_tmp)[0])+k] - candidate_tmp
                    feature_tmp = np.linalg.norm(difference)
                    if feature_tmp < dist:
                        dist = feature_tmp
                feature.append(dist)

        # turn list to array and reshape
        feature = np.asarray(feature)
        feature = feature.reshape(np.shape(X)[0], final_shapelet_num)

        return feature

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        return self.classifier.score(features, y)

class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Length of the compared positive and negative samples
           in the loss. Ignored if None, or if the time series in the training
           set have unequal lengths.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param epochs Number of epochs to run during the training of the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, batch_size=1, epochs=1000, lr=0.001,
                 channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, batch_size,
            epochs, lr,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        length = np.shape(X)[2]
        features = np.full(
            (np.shape(X)[0], self.out_channels, length), np.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # First applies the causal CNN
                output_causal_cnn = causal_cnn(batch)
                after_pool = torch.empty(
                    output_causal_cnn.size(), dtype=torch.double
                )
                if self.cuda:
                    after_pool = after_pool.cuda(self.gpu)
                after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                # Then for each time step, computes the output of the max
                # pooling layer
                for i in range(1, length):
                    after_pool[:, :, i] = torch.max(
                        torch.cat([
                            after_pool[:, :, i - 1: i],
                            output_causal_cnn[:, :, i: i+1]
                         ], dim=2),
                        dim=2
                    )[0]
                features[
                    count * batch_size: (count + 1) * batch_size, :, :
                ] = torch.transpose(linear(
                    torch.transpose(after_pool, 1, 2)
                ), 1, 2)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, batch_size, epochs, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size, epochs, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self
