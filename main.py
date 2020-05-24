import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
import json
import pandas as pd
import re
from flask_cors import CORS, cross_origin
import pickle
from newspaper import Article
import math
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from Bio import Entrez, Medline
from io import StringIO
import summariser

app = Flask(__name__)
cors = CORS(app)

# loading credibility scoring model
model = pickle.load(open('model.sav', 'rb'))
with open('tfidf_small.pickle', 'rb') as handle:
    tfidf = pickle.load(handle)

# loading summarizer model
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('bart-large-cnn')
SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
SUMMARY_MODEL.to(torch_device)
SUMMARY_MODEL.eval()
Entrez.email = 'pubmedemail@gmail.com'

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
        inp = request.get_json(force=True)
        print("old input")
        print(inp)
        print("new input")
        print(inp['url']['rawData'])
        #print("The result is {}.").format(inp)
        parsed_json = inp["url"]['rawData']
        result = []
        for raw_article in parsed_json:
            print("looping through individual article")
            print(raw_article)
            url = raw_article["url"]
            if "http" in url:
                article = Article(url, language="en")
            else:
                url = "http://" + url
                article = Article(url, language="en")
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
        print("results are")
        print(result)
        return jsonify(result)

    else:
        return redirect('http://localhost:5000')
        
@app.route('/predict', methods = ['GET', 'POST'])
#@cross_origin()
def predict():
    if request.method == 'POST':
        global model
        inp = request.data
        url = inp.decode('utf-8')
        if True:
            url = eval(url)
            print("url is")
            print(url)
            article = Article(url, language="en") # en for English 
            article.download()
            article.parse()
            title = article.title
            image = article.top_image
            description = article.text
            time = article.publish_date
            data = [article.text]
            article = pd.DataFrame({"text" : data}, index=[1])
            article['text'] = article['text'].apply(clean_text)
            X_newtesttext = tfidf.transform(article['text']).toarray()
            X_new = pd.DataFrame(X_newtesttext)
            pred = model.predict_proba(X_new)[0][0]*100
            credibility = math.floor(pred)
            result = {
                'title': title,
                'image': image,
                'description': description,
                'time': time,
                'sourceName': url.split('www.')[1].split('.com')[0],
                'sourceURL': url,
                'credibility': credibility
            }
            print(result)
            return jsonify(result)
        else:
            print("Please enter a link to the article starting with http")
            return "Please enter a link to the article starting with http"

    else:
        return redirect('http://localhost:5000')

@app.route('/sci', methods = ['GET', 'POST'])
#@cross_origin()
def sci():
    if request.method == 'POST':
        inp = request.data
        query = inp.decode('utf-8')
        results = summariser.pubMedSearch(query)
        results = pd.DataFrame(results).transpose()

        results['summary'] = results['abstract'].apply(summariser.getsummary)
        results['link'] = results['doi'].apply(summariser.getLink)
        results = results[['title', 'link', 'summary']]
        results_dict = results.to_dict('index')

        return jsonify(results_dict)

    else:
        return redirect('http://localhost:5000')

if __name__ == '__main__':
    app.run(debug = True)
