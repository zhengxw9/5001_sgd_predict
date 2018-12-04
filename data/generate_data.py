from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from datetime import datetime

def gen_duration(train_f):
    time_list = []
    class_para_keys = ['n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale' ]
    sgd_keys = ['penalty','l1_ratio','alpha','max_iter','random_state','n_jobs']
    for index, row in train_f.iterrows():
        para_dict = row.to_dict()
        sgd_dict = {}
        class_dict = {}
        for key in sgd_keys:
            sgd_dict[key] = para_dict[key]
        for key in class_para_keys:
            class_dict[key] = para_dict[key]
        start = datetime.now()
        X,y=make_classification(**class_dict)
        model = SGDClassifier(**sgd_dict)
        model.fit(X,y)
        duration = datetime.now()-start
        #print(row['time'],row['n_jobs'])
        time_list.append(duration.total_seconds())
    return time_list

train_f = pd.read_csv('./data/train.csv')
test_f = pd.read_csv('./data/test.csv')


for j in range(3):
	train_time_all = []
	test_time_all = []
	for k in range(10):
		print(k)
		train_time = gen_duration(train_f)
		train_time_all.append(train_time)
		test_time = gen_duration(test_f)
		test_time_all.append(test_time)
	train_time_pd = pd.DataFrame(train_time_all)
	train_time_pd.to_csv("./new_data/cpu8_mc_train_data"+str(j)+".csv")
	test_time_pd = pd.DataFrame(test_time_all)
	test_time_pd.to_csv("./new_data/cpu8_mc_test_data"+str(j)+".csv")
