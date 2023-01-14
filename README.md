# IMDB_Review_SentimentAnalysis

This project gives the viewers' sentiment of any movie, series, etc according to the most recent reviews given in the IMDB website. 

The model for sentiment analysis is trained using a BERT model in imdb_train.ipynb on the IMDB 50K reviews dataset. 25K reviews are taken from the dataset to train due to a heavy model and limited resources.

The app.py creates a website using flask and deploys the model. We scrap the user reviews from the IMDB website and do sentiment analysis on each of the reviews. Then we average the sentiment and classify it as "good" or "bad".

Various templates are provided to help build the website.

The link for the already trained model is: https://drive.google.com/file/d/1sgHL7pYxwYh43cCFt7H33V2DL6BRyyql/view?usp=share_link
