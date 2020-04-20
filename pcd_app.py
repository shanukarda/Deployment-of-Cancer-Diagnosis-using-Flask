# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:12:04 2020

@author: Shanu
"""
import re
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from collections import Counter
from scipy.sparse import hstack
import math
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
# model = pickle.load(open("pcd_app.pkl","rb"))

# loading ML model for prediction
with open("model_sig_clf.pkl", 'rb') as handle:
    sig_clf = pickle.load(handle)

# loading gene dictionary
with open("Gene_gv_dict.pkl", 'rb') as handle:
    gene_dict = pickle.load(handle)

# loading variant dictionary
with open("Variant_gv_dict.pkl", 'rb') as handle:
    variant_dict = pickle.load(handle)
    
# loading total word dictionary
with open("total_dict.pkl", 'rb') as handle:
    total_dict = pickle.load(handle)
    
# loading class wise word dictionary
with open("dict_list.pkl", 'rb') as handle:
    dict_list = pickle.load(handle)

@app.route("/")
def home():
    return render_template("index.html")

  
@app.route("/predict",methods =["POST"])
def predict():
    
    input_values = [x for x in request.form.values()]
    
    # gene feature encoding
    if input_values[0] in gene_dict.keys():
        gene_feature_rc = np.array(gene_dict[input_values[0]])
    else:
        gene_feature_rc = np.array([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
    
    # variant feature encoding
    if input_values[1] in variant_dict.keys():
        variant_feature_rc = np.array(variant_dict[input_values[1]])
    else:
        variant_feature_rc = np.array([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
    
    
    # text feature encoding
    if type(input_values[2]) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', input_values[2])
        # replace multiple spaces with single space
        total_text = re.sub('\s+',' ',total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
        
        text_fea_nlp_preprc = string
        
        
    
    text_fea_responseCoding = np.zeros((1,9))
    for i in range(0,9):
        sum_prob = 0
        for word in text_fea_nlp_preprc.split():
            sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))
    
    text_fea_responseCoding[0][i] = math.exp(sum_prob/len(text_fea_nlp_preprc.split()))
    
    text_fea_responseCoding = np.ravel(text_fea_responseCoding)
    
    input_responseCoding = np.hstack((gene_feature_rc,variant_feature_rc))
    input_responseCoding = np.hstack((input_responseCoding,text_fea_responseCoding))
    input_responseCoding = np.array([input_responseCoding])
    
    prediction = sig_clf.predict(input_responseCoding)[0]
    
    return render_template("index.html", 
                           prediction_text = "Given Gene, variant & text are belongs to Class {}".format(prediction))


if __name__ == "__main__":
    app.run(debug = True)