import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.datasets
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class MyLDA():
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.w = None
        self.priors = None
        self.gaussian_means = None
        self.gaussian_cov = None

    def fit(self, X_train, y_train):
        means = {}
        for c in range(self.num_classes):
            oneclass = X_train.loc[y_train == c]
            means[c] = np.array(oneclass.mean(axis=0))

        overall_mean = np.array(X_train.mean(axis=0))

        # Between class covariance matrix
        # Sb = sum {N_i (m_i - m) (m_i - m).T}
        ncol = X_train.shape[1]
        Sb = np.zeros((ncol, ncol))
        for c in range(self.num_classes):
            oneclass = X_train.loc[y_train == c]

            mat = np.outer((means[c] - overall_mean), (means[c] - overall_mean))
            Sb += np.multiply(len(oneclass), mat)

        # Within class covariance
        # Sw = sum {Si}
        # Si = sum {(x - m_i) (x - m_i).T}
        Sw = np.zeros(Sb.shape)
        for c in range(self.num_classes):
            oneclass = X_train.loc[y_train == c]
            tmp = np.subtract(oneclass.T, np.expand_dims(means[c], axis=1))
            Sw = np.add(np.dot(tmp, tmp.T), Sw)

        mat = np.dot(np.linalg.pinv(Sw), Sb)
        eigvals, eigvecs = np.linalg.eig(mat)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)
        self.w = np.array([eiglist[i][1] for i in range(self.num_classes)])

        # Parameters for predict()
        self.priors = {}
        self.gaussian_means = {}
        self.gaussian_cov = {}

        for c in range(self.num_classes):
            oneclass = X_train.loc[y_train == c]
            self.priors[c] = oneclass.shape[0] / float(X_train.shape[0])

            proj = np.dot(self.w, oneclass.T).T
            self.gaussian_means[c] = np.mean(proj, axis=0)
            self.gaussian_cov[c] = np.cov(proj, rowvar=False)

        return

    def predict(self, X_test):
        classes = range(self.num_classes)

        proj = np.dot(self.w, X_test.T).T
        # Likelihoods for each class
        likelihoods = np.array([[self.priors[c] * self.pdf([x[ind] for ind in range(len(x))],
                                self.gaussian_means[c], self.gaussian_cov[c])
                                for c in classes]
                                for x in proj])
        y_pred = np.argmax(likelihoods, axis=1)
        return y_pred

    def pdf(self, point, mean, cov):
        cons = 1. / ((2 * np.pi) ** (len(point) / 2.) * np.linalg.det(cov) ** (0.5))
        return cons * np.exp(np.dot(np.dot((point - mean), np.linalg.inv(cov)), (point - mean).T) * (-0.5))


def plot_2_hists(trainerr, testerr, min, max, binwidth):
    '''
    Plot 2 histograms on one picture
    '''
    bins = np.arange(min, max, binwidth)

    plt.hist(trainerr, bins, alpha=0.5, label='train')
    plt.hist(testerr, bins, alpha=0.5, label='test')
    plt.legend(loc='upper right')
    plt.show()


def split_into_train_test(data, split_ratio, ycol):
    traindata = []
    testdata = []

    grouped = data.groupby(data.ix[:, ycol])
    classes = [c for c in grouped.groups.keys()]

    classwise = {}
    for c in classes:
        classwise[c] = grouped.get_group(c)
        nrows_train = int(classwise[c].shape[0] * split_ratio)

        rows = random.sample(list(classwise[c].index), nrows_train)

        traindata.append(classwise[c].ix[rows])
        testdata.append(classwise[c].drop(rows))

    traindata = pd.concat(traindata)
    testdata = pd.concat(testdata)

    return traindata, testdata

def fit_get_errors(train, test, alg, ycol):
    '''
    Training and err counting on train and test sets
    '''

    y_train = train.target
    X_train = train.drop(data.columns[[ycol]], axis=1)

    alg.fit(X_train, y_train)

    y_train_pred = alg.predict(X_train)
    error_train = 1 - accuracy_score(y_train, y_train_pred)

    y_test = test.target
    X_test = test.drop(train.columns[[ycol]], axis=1)

    y_test_pred = alg.predict(X_test)
    error_test = 1 - accuracy_score(y_test, y_test_pred)

    return error_train, error_test


if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

    ############### Settings #################

    # Options: sk_lda, my_lda
    algorithm = "my_lda"

    ycol = 4  # label column number
    num_iter = 100  # number of experiments 
    binwidth = 0.005  #
    split_ratio = 0.7  # train set ratio

    # default alg
    alg = LinearDiscriminantAnalysis(store_covariance=True, solver='eigen')

    if algorithm == "my_lda":
        print("Using my LDA")
        alg = MyLDA(num_classes=3)
    else:
        print("Default: Using sklearn LDA")

    train_errors = np.zeros(num_iter)
    test_errors = np.zeros(num_iter)

    # 100 experiments
    for i in range(num_iter):
        # random split into train and test sets
        train, test = split_into_train_test(data, split_ratio, ycol)

        error_train, error_test = fit_get_errors(train, test, alg, ycol)

        train_errors[i] = error_train
        test_errors[i] = error_test
        print("train Error:", error_train)
        print("test Error:", error_test)
        print("-------------------")

    # Plot precision of classification (train and test)
    plot_2_hists(1-train_errors, 1-test_errors, 0.8, 1.1, binwidth)


