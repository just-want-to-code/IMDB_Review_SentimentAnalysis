# IMDB_Review_SentimentAnalysis

This project gives the most recent viewers' sentiment of any movie, series, etc according to the reviews given in the IMDB website. 

The model for sentiment analysis is trained using a BERT model in imdb_train.ipynb on the IMDB 50K reviews dataset. 25K reviews are taken from the dataset to train due to a heavy model and limited resources.

The review_inference.ipynb creates a website using flask and deploys the model. We scrap the user reviews from the IMDB website and do sentiment analysis on each of the reviews. Then we average the sentiment and classify it as "good" or "bad".

The index.html provides a template for the website.

The link for the already trained model is: https://drive.google.com/file/d/1sgHL7pYxwYh43cCFt7H33V2DL6BRyyql/view?usp=share_link
