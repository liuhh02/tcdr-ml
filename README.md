# Too Complicated, Didn't Read!

## Inspiration
Scientific literature too complicated to understand? Flooded with too many news articles? Use TC,DR! to simplify the information overload about COVID-19!

## Theme
Our project's theme is: Collecting, visualizing, and sharing information.

## What it does
Our web app has three components:
1. News Feed - Displays all relevant news articles relating to COVID-19, along with a credibility score calculated using a machine learning model trained from scratch to calculate the credibility of news articles

2. News Search Bar - Confused about whether to trust an article? Paste the article URL to find the credibility of the article!

3. Scientific Literature Search and Summary - Unsure about the current progress in the scientific literature? Uncomfortable with reading complicated research papers? Enter your query to receive links to relevant papers and a summary of the findings!

## How We built it
The web app is built using two machine learning models: firstly an xgboost model that has been trained on a dataset we collected for this hackathon to predict the credibility of news articles, and secondly the BART summarizer, a state-of-the-art seq2seq natural language processing model finetuned on scientific research papers that distils information from the scientific research papers into simple summaries that are easily understandable by audiences from all backgrounds.
1. For the news feed, we call the google news API to fetch news articles related to COVID. Then, we call the machine learning model to predict the credibility score of the article.
2. For the news search bar, we use the newspaper library to scrape the article from the link and obtain the text of the article. Then, we call on the machine learning model to predict the credibility score of the article.
3. For the scientific literature search and summary, we call the PubMed API to return scientific papers relating to the COVID query. Then, we use the BART summarizer to summarize the text of the scientific paper and display it along with the title and link of the paper.

## Impact of our Project
We hope our project can help clear up the large amount of uncertainty and unease surrounding the COVID-19 pandemic by keeping users informed about the latest developments through the news feed, while also helping them make more informed decisions as to whether to trust the article or not. Furthermore, the scientific literature search and summary aims to empower people who are do not have an extensive background in academia to easily understand what is going on in the current research community. This will help prepare us for the future, to fight against misinformation and empower people to take charge of their situation.

## Challenges I ran into
1. We had issues connecting a **Flask** Backend to a **React** Frontend due to the very little documentation available. This took up most of our time.
2. Since we are a two-people team, it was difficult to accomplish so many things: coding the frontend, backend, training the ML models, and finally connecting the Flask backend to the React frontend.

## Accomplishments that we are proud of
We are proud of giving something back to the tech community that helps us with the smallest bugs we have to the biggest technology we want to learn. We are also proud to be helpful to help in this fight with the covid-19 pandemic. 

## What we learned
We learnt to perform so many tasks in a short period of time, being in different timezones, it was even difficult for us to collaborate but we managed really well.

## What's next for TC,DR! - Too Complicated Didn't Read
We hope to deploy the website online so it is accessible to everyone to use. 

## How to go about our project
There are two github repositories.
One having the frontend, when started runs on port:3000.
The other having the code for backend and the flask server, which when run, runs on port:5000

This repository contains two machine learning models:
1. `model.sav` contains a classifier trained on 10000 samples of real news and 10000 samples of fake news. Provide the text of the news article and the model will output the predicted credibility of the news article. The tfidf tokenizer is used and is named `tfidf_small.pickle`.
2. The second model is the [BART summarizer](https://arxiv.org/abs/1910.13461) that summarizes scientific abstracts and outputs a description that is easy to understand.

The repository for the React frontend can be found [here](https://github.com/r-ush/tcdr-frontend).

For more information, check the Devpost submission [here](https://devpost.com/software/tcdr).
