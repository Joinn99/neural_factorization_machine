'''
Utilities for Loading data.
The input data file follows the same input for LibFM: http://www.libfm.org/libfm-1.42.manual.pdf

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@references:
'''
import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

class LoadData(object):
    '''given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset +".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = self.map_features( )
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        #print("features_M:", len(self.features))
        return  len(self.features)

    def read_features(self, file): # read a feature file
        f = open( file )
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.features:
                    self.features[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        print("# of training:" , len(Y_))

        X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        print("# of validation:", len(Y_))

        X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        print("# of test:", len(Y_))

        return Train_data,  Validation_data,  Test_data

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open( file )
        X_ = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )
            line = f.readline()
        f.close()
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = np.array([ Y_[i] for i in indexs])
        Data_Dic['X'] = np.array([ X_[i] for i in indexs])
        return Data_Dic
    
    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in xrange(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in xrange(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in xrange(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in xrange(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable


class LoadMat(object):
    '''given the path of *.mat data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''
    def __init__(self, path, dataset, loss_type):
        self.path = path + dataset + ".mat"
        self.loss_type = loss_type
        self.construct_data(5, 1)

    def _libFM_format(self, Data, Ind, sizel):
        rep_is = np.repeat(Data["is"][Ind, :], sizel, axis=0)
        rep_iu = np.repeat(Data["iu"][Ind, :], sizel, axis=0)
        rep_tag = np.tile(np.arange(sizel), Ind.shape[0]).reshape(-1, 1)
        return np.concatenate([rep_is, rep_iu, rep_tag], axis=1)


    def construct_data(self, CV, TI):
        Data = loadmat(self.path)
        sizeN = Data["E"].shape[0]
        ITrain = np.where(np.arange(1, sizeN+1) % CV != TI)[0]
        ITest = np.where(np.arange(1, sizeN+1) % CV == TI)[0]
        sizem = np.max(Data["is"]) + 1
        sizen = np.max(Data["iu"]) + 1
        sizel = Data["E"].shape[1]
        self.features_M = sizem + sizen + sizel

        Data["is"] = Data["is"] + sizel
        Data["iu"] = Data["iu"] + sizel + sizem
        
        if self.loss_type == "square_loss":
            Data["E"] = np.sign(Data["E"] - 0.5)
        else:
            Data["E"] = Data["E"]

        self.Train_data = {}
        self.Validation_data = {}
        self.Train_data["X"], self.Validation_data["X"], self.Train_data["Y"], self.Validation_data["Y"] = train_test_split(
            self._libFM_format(Data, ITrain, sizel), Data["E"][ITrain, :].reshape(-1, 1), test_size=0.22)
        self.Test_data = {"X": self._libFM_format(Data, ITest, sizel), "Y": Data["E"][ITest, :].reshape(-1, 1)}

        self.Train_data["Y"] = self.Train_data["Y"].flatten()
        self.Validation_data["Y"] = self.Validation_data["Y"].flatten()
        self.Test_data["Y"] = self.Test_data["Y"].flatten()