## Too Complicated, Didn't Read!

This repository was created as part of the [OpenHacks hackathon](https://openhacks.devpost.com/). The web app has three components:

1. News Feed - Displays all relevant news articles relating to COVID-19, along with a credibility score calculated using a machine learning model trained from scratch to calculate the credibility of news articles.

2. News Search Bar - Confused about whether to trust an article? Paste the article URL to find the credibility of the article!

3. Scientific Literature Search and Summary - Unsure about the current progress in the scientific literature? Uncomfortable with reading complicated research papers? Enter your query to receive links to relevant papers and a summary of the findings!

This repository contains two machine learning models:
1. `model.sav` contains a classifier trained on 10000 samples of real news and 10000 samples of fake news. Provide the text of the news article and the model will output the predicted credibility of the news article. The tfidf tokenizer is used and is named `tfidf_small.pickle`.
2. The second model is the [BART summarizer](https://arxiv.org/abs/1910.13461) that summarizes scientific abstracts and outputs a description that is easy to understand.

The repository for the React frontend can be found [here](https://github.com/r-ush/tcdr-frontend).

For more information, check the Devpost submission [here](https://devpost.com/software/tcdr).
