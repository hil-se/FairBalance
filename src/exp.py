from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from fermi import FERMI
from load_data import load
import numpy as np
from preprocessor import Reweighing, FairBalance, FairBalanceVariant
from metrics import Metrics

class Exp:
    def __init__(self, data, treatment="None", inject = None):
        #  Load data
        self.data, self.A = load(data)
        # Separate independent variables and dependent variables
        independent = self.data.keys().tolist()
        dependent = independent.pop(-1)
        self.X = self.data[independent]
        self.y = np.array(self.data[dependent])
        self.treatment = treatment
        self.inject = inject
        if treatment == "FERMI":
            self.clf = FERMI()
        else:
            self.clf = LogisticRegression(max_iter=100000)

    def one_exp(self):
        X_train, X_test, y_train, y_test = self.train_test_split(test_size=0.5)
        #########################################
        self.data_preprocess(X_train)
        sample_weight = self.treat(X_train, y_train)
        self.fit(X_train, y_train, sample_weight)
        m_train = Metrics(self.clf, X_train, y_train, self.A, self.preprocessor)
        m_test = Metrics(self.clf, X_test, y_test, self.A, self.preprocessor)
        return m_train, m_test

    def fit(self, X, y, sample_weight=None):
        X_train_processed = self.preprocessor.fit_transform(X)
        if type(self.clf) == FERMI:
            S = []
            groups = {}
            count = 0
            for i in range(len(y)):
                group = tuple([X[a][i] for a in self.A])
                if group not in groups:
                    groups[group] = count
                    count += 1
                S.append(groups[group])
            S = np.array(S)
            self.clf.fit(X_train_processed, y, S, sample_weight=sample_weight)
        else:
            self.clf.fit(X_train_processed, y, sample_weight=sample_weight)

    def data_preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown = 'ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])

    def treat(self, X_train, y_train):
        if self.treatment == "Reweighing":
            sample_weight = Reweighing(X_train, y_train, self.A)
        elif self.treatment == "FairBalanceVariant":
            sample_weight = FairBalanceVariant(X_train, y_train, self.A)
        elif self.treatment == "FairBalance":
            sample_weight = FairBalance(X_train, y_train, self.A)
        else:
            sample_weight = None
        return sample_weight

    def train_test_split(self, test_size=0.5):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.y)):
            key = tuple([self.X[a][i] for a in self.A] + [self.y[i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key])*test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)
        X_train = self.X.iloc[train]
        X_test = self.X.iloc[test]
        y_train = self.y[train]
        y_test = self.y[test]
        X_train.index = range(len(X_train))
        X_test.index = range(len(X_test))
        return X_train, X_test, y_train, y_test

