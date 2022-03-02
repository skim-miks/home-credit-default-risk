import pandas as pd
import pickle
import ast
import json

def to_string(data):
	cat_columns = [col for col in data.columns if data[col].dtype == 'object']
	data[cat_columns] = data[cat_columns].astype(str)
	return data


def json_to_df(string_to_df):
	x = pd.DataFrame(string_to_df, index=[0])

	print('json successfully converted to data frame, x')
	return x


def load_pickle(str_filename='../06_preprocessing/output/dict_imputations.pkl'):
	# import pickled_file
	pickled_file = pickle.load(open(str_filename, 'rb'))
	# print message
	print(f'Imported {str_filename}')
	# return pickled_file
	return pickled_file

def output(int_id, yhat):
  	df_output = pd.DataFrame({'Row_id': [int_id],
                            'Score': [yhat]})
  	str_output = df_output.to_json(orient='records')
  	list_output = ast.literal_eval(str_output)
  
  	output_final = {"Request_id":"",
                  	"Zaml_processing_id":"",
                  	"Response":[{"Model_name":"interview-Model",
                               	"Model_version":"v1",
                               	"Results":list_output}]}
  	return output_final

def model_api(string_to_df, feat_list, cb_model):
  	x = json_to_df(string_to_df)
  
  	int_id = x['SK_ID_CURR'].iloc[0]
  	x = x[feat_list]
  
  	x = to_string(x)
  
  	yhat = cb_model.predict_proba(x)[0,1]
  	
  	output_final  = output(int_id, yhat)
  	return output_final


