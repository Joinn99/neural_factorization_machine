import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

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

        self.Train_data["Y"] = self.Train_data["Y"].T
        self.Validation_data["Y"] = self.Validation_data["Y"].T
        self.Test_data["Y"] = self.Test_data["Y"].T




