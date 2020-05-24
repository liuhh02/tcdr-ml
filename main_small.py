import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
import json
import pandas as pd
import re
from flask_cors import CORS, cross_origin
import pickle
from newspaper import Article
import math
#from transformers import BartTokenizer, BartForConditionalGeneration
#import torch
#from Bio import Entrez, Medline
#from io import StringIO
#import summariser

app = Flask(__name__)
cors = CORS(app)

# loading credibility scoring model
model = pickle.load(open('model.sav', 'rb'))
with open('tfidf_small.pickle', 'rb') as handle:
    tfidf = pickle.load(handle)

# # loading summarizer model
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('bart-large-cnn')
# SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
# SUMMARY_MODEL.to(torch_device)
# SUMMARY_MODEL.eval()
# Entrez.email = 'pubmedemail@gmail.com'

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
        # inp = request.data
        # query = inp.decode('utf-8')
        # results = summariser.pubMedSearch(query)
        # results = pd.DataFrame(results).transpose()

        # results['summary'] = results['abstract'].apply(summariser.getsummary)
        # results['link'] = results['doi'].apply(summariser.getLink)
        # results = results[['title', 'link', 'summary']]
        results_dict = 
        {'pm_32437587': {'title': 'COVID-19 vaccines: knowing the unknown.', 'link': 'http://dx.doi.org/10.1002/eji.202048663', 'summary': 'COVID-19 is caused by a new coronavirus, SARS-CoV-2. Previous research on other coronavirus vaccines, such as FIPV, SARS and MERS, has provided valuable information. We are therefore optimistic about the rapid development of COVID-19 vaccine.'}, 'pm_32433465': {'title': 'Immunogenicity of a DNA vaccine candidate for COVID-19.', 'link': 'http://dx.doi.org/10.1038/s41467-020-16505-0', 'summary': 'The coronavirus family member, SARS-CoV-2 has been identified as the causal agent for the pandemic viral pneumonia disease, COVID-19. At this time, no vaccine is available to control further dissemination of the disease. We have previously engineered a synthetic DNA vaccine targeting the MERS coronavirus Spike (S) protein, the major surface antigen of coronaviruses.'}, 'pm_32425000': {'title': 'An overview of COVID-19.', 'link': 'http://dx.doi.org/10.1631/jzus.B2000083', 'summary': 'Pneumonia caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection emerged in Wuhan City, Hubei Province, China in December 2019. Wild animal hosts and infected patients are currently the main sources of disease which is transmitted via respiratory droplets and direct contact.'}, 'pm_32422384': {'title': 'Diagnosis and treatment of coronavirus disease 2019 (COVID-19): Laboratory, PCR, and chest CT imaging findings.', 'link': 'http://dx.doi.org/10.1016/j.ijsu.2020.05.018', 'summary': 'Since December 2019, more than 3 million cases of coronavirus disease 2019 (COVID-19) and about 200,000 deaths have been reported worldwide. About 5% of cases were considered critically ill and 14% were considered to have the severe classification of the disease. In China, the fatality rate of this infection was about 4%.'}, 'pm_32418793': {'title': "An effective CTL peptide vaccine for Ebola Zaire Based on Survivors' CD8+ targeting of a particular nucleocapsid protein epitope with potential implications for COVID-19 vaccine design.", 'link': 'http://dx.doi.org/10.1016/j.vaccine.2020.04.034', 'summary': 'The 2013-2016 West Africa EBOV epidemic was the biggest EBOV outbreak to date. An analysis of virus-specific CD8+ T-cell immunity in 30 survivors showed that 26 of those individuals had a CD8+ response to at least one EBOV protein. A single vaccination in a C57BL/6 mouse using an adjuvanted microsphere peptide vaccine formulation containing NP44-52 is enough to confer immunity in mice.'}, 'pm_32413300': {'title': 'SnapShot: COVID-19.', 'link': 'http://dx.doi.org/10.1016/j.cell.2020.04.013', 'summary': 'Coronavirus disease 2019 (COVID-19) is a novel respiratory illness caused by SARS-CoV-2. Most cases are mild; severe disease often involves cytokine storm and organ failure. Therapeutics including antivirals, immunomodulators, and vaccines are in development.'}, 'pm_32407707': {'title': 'Rational Vaccine Design in the Time of COVID-19.', 'link': 'http://dx.doi.org/10.1016/j.chom.2020.04.022', 'summary': 'As scientists consider SARS-CoV-2 vaccine design, we discuss problems that may be encountered and how to tackle them. We draw on experiences from recent research on several viruses including HIV and influenza, as well as coronaviruses. We further discuss approaches to pan-coronavirus vaccines.'}, 'pm_32407706': {'title': 'SARS-CoV-2: A New Song Recalls an Old Melody.', 'link': 'http://dx.doi.org/10.1016/j.chom.2020.04.019', 'summary': 'The viruses causing the SARS outbreak of 2002-2003 and current COVID-19 pandemic are related betacoronaviruses. Focusing on important lessons from SARS vaccine development and two SARS vaccines evaluated in humans may guide SARS-CoV-2 vaccine design, testing, and implementation.'}}
        #results_dict = results.to_dict('index')

        return jsonify(results_dict)

    else:
        return redirect('http://localhost:5000')

if __name__ == '__main__':
    app.run(debug = True)
