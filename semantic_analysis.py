import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob

#loading in necessary spacy packages like english language package and spacytextblob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

#loading in dataset as a table
df = pd.read_csv("Amazon_Product_reviews.csv")
df.head()

#removing data entries without a review and picking only the reviews.text column
reviews_data_clean = df.dropna(subset=["reviews.text"])
reviews_data = reviews_data_clean["reviews.text"]

#defining a function to clean the chosen review for better processing
def clean_data(Review):
    print(Review)
    Review_new= Review.lower()                #ensuring the review string is all lowercase
    Review_new= Review_new.strip()            #removing blank spaces from the strings
    return Review_new


#defining a function for analysing the sentiment of the cleaned review 
def sentiment(string):
    Review_new = clean_data (string)
    doc = nlp(Review_new)
    doc_list = [token.orth_ for token in doc if not token.is_stop]                    # removing the stop words to leave only key parts of sentence for analysis
    doc_new = " ".join(doc_list)                                                      #recreating string of key parts of sentence
    doc = nlp(doc_new)                                                                #retokenising the new string without stop words for nlp processing
    print(doc)
    #polarity = doc._.blob.polarity
    sentiment = doc._.blob.sentiment                                                  #finding the sentiment of the review
    print(sentiment)
    print(doc._.blob.sentiment_assessments.assessments)                           #detailing how each word token was assessed

#assigning positiv/negative labels to the polarity of the sentiment function - picking the polarity only
    if sentiment[0] > 0:
        sent_label = "positive"
    elif sentiment[0] < 0:
        sent_label = "negative"
    else:
        sent_label = "neutral"                   #else case that though unlikely is included in the event that the polarity is exactly 0
    return sent_label


#picking out sample reviews for testing
First_review = reviews_data[0]
Second_review = reviews_data[3984]
third_review = reviews_data[2897]
fourth_review = reviews_data[422]
fifth_review = reviews_data[289]
sixth_review = reviews_data[1573]
seventh_review = reviews_data[4237]
eighth_review = reviews_data[703]
ninth_review = reviews_data[1206]
tenth_review = reviews_data[4369]


#analysing the sentiment of the sample reviews
first_clean = clean_data(First_review)
first_sent = sentiment(first_clean)
print("Sentiment One =", first_sent)

print("--------------")

second_clean = clean_data(Second_review)
second_sent = sentiment(second_clean)
print("Sentiment two =", second_sent)

print("--------------")

third_clean = clean_data(third_review)
third_sent = sentiment(third_clean)
print("sentiment three =", third_sent)

print("--------------")

fourth_clean = clean_data(fourth_review)
fourth_sent = sentiment(fourth_clean)
print("sentiment four =", fourth_sent)

print("--------------")

fifth_clean = clean_data(fifth_review)
fifth_sent = sentiment(fifth_clean)
print("sentiment five =", fifth_sent)

print("--------------")

sixth_clean = clean_data(sixth_review)
sixth_sent = sentiment(sixth_clean)
print("sentiment six =", sixth_sent)

print("--------------")

seventh_clean = clean_data(seventh_review)
seventh_sent = sentiment(seventh_clean)
print("sentiment seven =", seventh_sent)

print("--------------")

eighth_clean = clean_data(eighth_review)
eighth_sent = sentiment(eighth_clean)
print("sentiment eight =", eighth_sent)

print("--------------")

ninth_clean = clean_data(ninth_review)
ninth_sent = sentiment(ninth_clean)

print("sentiment nine =", ninth_sent)

print("--------------")

tenth_clean = clean_data(tenth_review)
tenth_sent = sentiment(tenth_clean)
print("sentiment ten =", tenth_sent)





    



