from functions import load_pickle, to_string, json_to_df, output, model_api
from flask import Flask, request, jsonify
import traceback

#with open('application_test_row_3.json') as f:
#  string_to_df = json.load(f)
  
app = Flask(__name__)

try:
  feat_list = load_pickle(str_filename='feat_list.pkl')
  
  cb = load_pickle(str_filename='cb_model.sav')
except:
  print('Error loading pickle')

@app.route('/', methods=['POST'])
def predict():
  try:
    string_to_df = request.get_json()
    
    out = model_api(string_to_df, feat_list, cb)
    
    return out
  
  except:
    
    return jsonify({'trace': traceback.format_exc()})
  
if __name__ == '__main__':
  
  app.run(debug=True)
