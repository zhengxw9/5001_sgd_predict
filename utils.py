import pandas as pd
import numpy as np

def transpose(df,prefix):
    df_dict = {}
    for index, row in df.iterrows():
        df_dict[prefix+str(index)] = row.values[1:]
    new_df = pd.DataFrame(df_dict)
    new_df[prefix+'median'] = np.median(new_df.values,axis=1)
    new_df[prefix+'max'] = np.max(new_df.values,axis=1)
    new_df[prefix+'min'] = np.min(new_df.values,axis=1)
    return new_df

def add_attrs_from_dict(df_all,**kwargs):
    for key, value in kwargs.items():
        new_df = transpose(value,key)
        df_all[key+'median'] = new_df[key+'median']
        df_all[key+'min'] = new_df[key+'min']
        df_all[key+'max'] = new_df[key+'max']

#the follwing functions is used to preprocess the dataframe
def one_hot(df,attr):
    df[attr] = pd.factorize(df[attr])[0]
    dummies_df = pd.get_dummies(df[attr], prefix=df[[attr]].columns[0])
    df = pd.concat([df, dummies_df], axis=1)
    df = df.drop(attr, axis=1)
    return df
	
def get_train_test_df():
	train_f = pd.read_csv('./data/train.csv')
	test_f = pd.read_csv('./data/test.csv')
	cpu12_train_all0 = pd.read_csv('./new_data/cpu12_all_train_data0.csv')
	cpu12_train_all1 = pd.read_csv('./new_data/cpu12_all_train_data1.csv')
	cpu12_test_all0 = pd.read_csv('./new_data/cpu12_all_test_data0.csv')
	cpu12_test_all1 = pd.read_csv('./new_data/cpu12_all_test_data1.csv')
	cpu12_train_mc = pd.read_csv('./new_data/cpu12_mc_train_data0.csv')
	cpu12_test_mc = pd.read_csv('./new_data/cpu12_mc_test_data0.csv')
	cpu8_train_all = pd.read_csv('./new_data/cpu8_train_data0.csv')
	cpu8_test_all = pd.read_csv('./new_data/cpu8_test_data0.csv')

	#one-hot encode the penalty attrbute
	train_f['train'] = 1
	test_f['train'] = 0
	data_all = pd.concat([train_f,test_f])
	data_all = one_hot(data_all,'penalty')
	train_f = data_all.loc[data_all['train'] == 1]
	test_f = data_all.loc[data_all['train'] == 0]
	train_f.drop('train',axis=1)
	test_f.drop('train',axis=1)

	train_prefix_dict = {'gen12_all0':cpu12_train_all0,'gen12_all1':cpu12_train_all1,'gen12_mc':cpu12_train_mc,'gen8_all':cpu8_train_all}
	test_prefix_dic = {'gen12_all0':cpu12_test_all0,'gen12_all1':cpu12_test_all1,'gen12_mc':cpu12_test_mc,'gen8_all':cpu8_test_all}
	add_attrs_from_dict(train_f,**train_prefix_dict)
	add_attrs_from_dict(test_f,**test_prefix_dic)
	
	return train_f,test_f

