import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
import json
#from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from flask_cors import CORS, cross_origin
import pickle
from newspaper import Article
import math

app = Flask(__name__)
cors = CORS(app)
# loading models
model = pickle.load(open('model.sav', 'rb'))
with open('tfidf_small.pickle', 'rb') as handle:
    tfidf = pickle.load(handle)
        
# loading similarity ML model
#model2 = SentenceTransformer('bert-base-nli-mean-tokens')
    
def clean_text(text):
    clean = re.sub(r'[^\x00-\x7f]',r'', text)
    regex = re.compile(r'[\n\r\t]')
    clean = regex.sub(" ", clean)
    return clean
	
@app.route('/load', methods = ['GET', 'POST'])
#@cross_origin()
def load():
    if request.method == 'POST':
        global model
        inp = request.rawData
        parsed_json = (json.loads(inp))
        result = []
        for raw_article in parsed_json['articles']:
            url = raw_article["url"]
            article = Article(url, language="en") # en for English 
            article.download()
            article.parse()
            data = [article.text]
            article = pd.DataFrame({"text" : data}, index=[1])
            article['text'] = article['text'].apply(clean_text)
            X_newtesttext = tfidf.transform(article['text']).toarray()
            X_new = pd.DataFrame(X_newtesttext)
            pred = model.predict_proba(X_new)[0][0]*100
            credibility = math.floor(pred)
            result.append(
                {
                    'title': raw_article["title"],
                    'image': raw_article["image"],
                    'description': raw_article["description"],
                    'time': raw_article["publishedAt"],
                    'sourceName': raw_article["source"]["name"],
                    'sourceURL': url,
                    'credibility': credibility
                }
            )
        return jsonify(result)

    else:
        return redirect('http://localhost:3000')
		
@app.route('/predict', methods = ['GET', 'POST'])
#@cross_origin()
def predict():
    if request.method == 'POST':
        global model
        inp = request
        print('postreq')
        print(inp)
        if True:
            article = Article(inp, language="en") # en for English 
            article.download()
            article.parse()
            data = [article.text]
            article = pd.DataFrame({"text" : data}, index=[1])
            article['text'] = article['text'].apply(clean_text)
            X_newtesttext = tfidf.transform(article['text']).toarray()
            X_new = pd.DataFrame(X_newtesttext)
            pred = model.predict_proba(X_new)[0][0]*100
            credibility = math.floor(pred)
            result = {
                'title': article.title,
                'image': article.top_image,
                'description': article.text,
                'time': article.publish_date,
                'sourceName': inp.split('www.')[1].split('.com')[0],
                'sourceURL': inp,
                'credibility': credibility
            }
            return jsonify(result)
        else:
            print("Please enter a link to the article starting with http")
            return "Please enter a link to the article starting with http"

    else:
        return redirect('http://localhost:3000')

if __name__ == '__main__':
    app.run(debug = True)
