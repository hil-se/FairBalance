from __future__ import print_function, division
from collections import Counter
from scipy.spatial import distance as dist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as NN
from aif360.datasets import StandardDataset

import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os

import sys
import pdb
import unittest
import random


class FairsmoteMultiple:

    def __init__(self, df, df_name):
        self.df = df
        self.df_name = df_name

    def run_fairsmote_multiple(self):

        dataset_orig_train = self.df.convert_to_dataframe()[0]

        if self.df_name == "adult":

            protected_attribute1 = "sex"
            protected_attribute2 = "race"

            # first one is class value and second one is 'sex' and third one is 'race'
            zero_zero_zero = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            zero_zero_one = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            zero_one_zero = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            zero_one_one = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            one_zero_zero = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            one_zero_one = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            one_one_zero = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            one_one_one = len(dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])

            maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)


            zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
            zero_zero_one_to_be_incresed = maximum - zero_zero_one
            zero_one_zero_to_be_incresed = maximum - zero_one_zero
            zero_one_one_to_be_incresed = maximum - zero_one_one
            one_zero_zero_to_be_incresed = maximum - one_zero_zero
            one_zero_one_to_be_incresed = maximum - one_zero_one
            one_one_zero_to_be_incresed = maximum - one_one_zero
            one_one_one_to_be_incresed = maximum - one_one_one


            df_zero_zero_zero = dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                       & (dataset_orig_train[protected_attribute2] == 0)]
            df_zero_zero_one = dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_zero_one_zero = dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_zero_one_one = dataset_orig_train[(dataset_orig_train['Income Binary'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_one_zero_zero = dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_one_zero_one = dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_one_one_zero = dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_one_one_one = dataset_orig_train[(dataset_orig_train['Income Binary'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]


            df_zero_zero_zero = self.generate_samples(zero_zero_zero_to_be_incresed,df_zero_zero_zero,'Adult')
            df_zero_zero_one = self.generate_samples(zero_zero_one_to_be_incresed,df_zero_zero_one,'Adult')
            df_zero_one_zero = self.generate_samples(zero_one_zero_to_be_incresed,df_zero_one_zero,'Adult')
            df_zero_one_one = self.generate_samples(zero_one_one_to_be_incresed,df_zero_one_one,'Adult')
            df_one_zero_zero = self.generate_samples(one_zero_zero_to_be_incresed,df_one_zero_zero,'Adult')
            df_one_zero_one = self.generate_samples(one_zero_one_to_be_incresed,df_one_zero_one,'Adult')
            df_one_one_zero = self.generate_samples(one_one_zero_to_be_incresed,df_one_one_zero,'Adult')
            df_one_one_one = self.generate_samples(one_one_one_to_be_incresed,df_one_one_one,'Adult')


            df = pd.concat([df_zero_zero_zero,df_zero_zero_one,df_zero_one_zero,df_zero_one_one, df_one_zero_zero,df_one_zero_one,df_one_one_zero,df_one_one_one])

            df['race'] = df['race'].astype(float)
            df['sex'] = df['sex'].astype(float)

            self.df = df

            self.df = StandardDataset( df=df, label_name='Income Binary', protected_attribute_names=['sex','race'], favorable_classes=[1],
                    privileged_classes=[[1]])

        if self.df_name == "compas":

            protected_attribute1 = "sex"
            protected_attribute2 = "race"


            # first one is class value and second one is 'sex' and third one is 'race'
            zero_zero_zero = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            zero_zero_one = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            zero_one_zero = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            zero_one_one = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            one_zero_zero = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            one_zero_one = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            one_one_zero = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            one_one_one = len(dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])

            maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)


            zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
            zero_zero_one_to_be_incresed = maximum - zero_zero_one
            zero_one_zero_to_be_incresed = maximum - zero_one_zero
            zero_one_one_to_be_incresed = maximum - zero_one_one
            one_zero_zero_to_be_incresed = maximum - one_zero_zero
            one_zero_one_to_be_incresed = maximum - one_zero_one
            one_one_zero_to_be_incresed = maximum - one_one_zero
            one_one_one_to_be_incresed = maximum - one_one_one


            df_zero_zero_zero = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_zero_zero_one = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_zero_one_zero = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_zero_one_one = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_one_zero_zero = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_one_zero_one = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_one_one_zero = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_one_one_one = dataset_orig_train[(dataset_orig_train['two_year_recid'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]


            df_zero_zero_zero = self.generate_samples(zero_zero_zero_to_be_incresed,df_zero_zero_zero,'Compas')
            df_zero_zero_one = self.generate_samples(zero_zero_one_to_be_incresed,df_zero_zero_one,'Compas')
            df_zero_one_zero = self.generate_samples(zero_one_zero_to_be_incresed,df_zero_one_zero,'Compas')
            df_zero_one_one = self.generate_samples(zero_one_one_to_be_incresed,df_zero_one_one,'Compas')
            df_one_zero_zero = self.generate_samples(one_zero_zero_to_be_incresed,df_one_zero_zero,'Compas')
            df_one_zero_one = self.generate_samples(one_zero_one_to_be_incresed,df_one_zero_one,'Compas')
            df_one_one_zero = self.generate_samples(one_one_zero_to_be_incresed,df_one_one_zero,'Compas')
            df_one_one_one = self.generate_samples(one_one_one_to_be_incresed,df_one_one_one,'Compas')


            df = pd.concat([df_zero_zero_zero,df_zero_zero_one,df_zero_one_zero,df_zero_one_one, df_one_zero_zero,df_one_zero_one,df_one_one_zero,df_one_one_one])
            df['race'] = df['race'].astype(float)
            df['sex'] = df['sex'].astype(float)

            self.df = df


            self.df = StandardDataset( df=df, label_name='two_year_recid', protected_attribute_names=['sex','race'], favorable_classes=[0],
                    privileged_classes=[[1]])

        if self.df_name == "german":

            protected_attribute1 = "sex"
            protected_attribute2 = "age"

            # first one is class value and second one is 'sex' and third one is 'race'
            zero_zero_zero = len(dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            zero_zero_one = len(dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            zero_one_zero = len(dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            zero_one_one = len(dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            one_zero_zero = len(dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            one_zero_one = len(dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])
            one_one_zero = len(dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)])
            one_one_one = len(dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)])

            maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)


            zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
            zero_zero_one_to_be_incresed = maximum - zero_zero_one
            zero_one_zero_to_be_incresed = maximum - zero_one_zero
            zero_one_one_to_be_incresed = maximum - zero_one_one
            one_zero_zero_to_be_incresed = maximum - one_zero_zero
            one_zero_one_to_be_incresed = maximum - one_zero_one
            one_one_zero_to_be_incresed = maximum - one_one_zero
            one_one_one_to_be_incresed = maximum - one_one_one


            df_zero_zero_zero = dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_zero_zero_one = dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_zero_one_zero = dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_zero_one_one = dataset_orig_train[(dataset_orig_train['credit'] == 2) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_one_zero_zero = dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_one_zero_one = dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]
            df_one_one_zero = dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 0)]
            df_one_one_one = dataset_orig_train[(dataset_orig_train['credit'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                                   & (dataset_orig_train[protected_attribute2] == 1)]


            df_zero_zero_zero = self.generate_samples(zero_zero_zero_to_be_incresed,df_zero_zero_zero,'German')
            df_zero_zero_one = self.generate_samples(zero_zero_one_to_be_incresed,df_zero_zero_one,'German')
            df_zero_one_zero = self.generate_samples(zero_one_zero_to_be_incresed,df_zero_one_zero,'German')
            df_zero_one_one = self.generate_samples(zero_one_one_to_be_incresed,df_zero_one_one,'German')
            df_one_zero_zero = self.generate_samples(one_zero_zero_to_be_incresed,df_one_zero_zero,'German')
            df_one_zero_one = self.generate_samples(one_zero_one_to_be_incresed,df_one_zero_one,'German')
            df_one_one_zero = self.generate_samples(one_one_zero_to_be_incresed,df_one_one_zero,'German')
            df_one_one_one = self.generate_samples(one_one_one_to_be_incresed,df_one_one_one,'German')

            
            df = pd.concat([df_zero_zero_zero,df_zero_zero_one,df_zero_one_zero,df_zero_one_one, df_one_zero_zero,df_one_zero_one,df_one_one_zero,df_one_one_one])
            df['sex'] = df['sex'].astype(float)
            df['age'] = df['age'].astype(float)

            self.df = df

            self.df = StandardDataset( df=df, label_name='credit', protected_attribute_names=['sex','age'], favorable_classes=[1],
                    privileged_classes=[[1]])

        return self.df

    def get_ngbr(self, df, knn):
            rand_sample_idx = random.randint(0, df.shape[0] - 1)
            parent_candidate = df.iloc[rand_sample_idx]
            ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=False)
            candidate_1 = df.iloc[ngbr[0][0]]
            candidate_2 = df.iloc[ngbr[0][1]]
            candidate_3 = df.iloc[ngbr[0][2]]
            return parent_candidate,candidate_2,candidate_3

    def generate_samples(self,no_of_samples,df,df_name):
        
        total_data = df.values.tolist()
        knn = NN(n_neighbors=5,algorithm='auto').fit(df)
        
        for _ in range(no_of_samples):
            cr = 0.8
            f = 0.8
            parent_candidate, child_candidate_1, child_candidate_2 = self.get_ngbr(df, knn)
            new_candidate = []
            for key,value in parent_candidate.items():
                if isinstance(parent_candidate[key], bool):
                    new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
                elif isinstance(parent_candidate[key], str):
                    new_candidate.append(random.choice([parent_candidate[key],child_candidate_1[key],child_candidate_2[key]]))
                elif isinstance(parent_candidate[key], list):
                    temp_lst = []
                    for i, each in enumerate(parent_candidate[key]):
                        temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                        int(parent_candidate[key][i] +
                                            f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                    new_candidate.append(temp_lst)
                else:
                    new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))        
            total_data.append(new_candidate)
        
        final_df = pd.DataFrame(total_data)

        columns = list(df.columns)
        for i in range(len(columns)):
            final_df = final_df.rename(columns={i: columns[i]}, errors="raise")

        return final_df

