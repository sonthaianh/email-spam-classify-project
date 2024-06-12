################ import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

################ import dataset
pathname_dataset = "Dataset\spam.csv"
spam_df = pd.read_csv(pathname_dataset)

################ inspect data
spam_df_group = spam_df.groupby('Category').describe()

################ label [1 is spam, 0 is not spam] 
spam_df_check_label = spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

################ create train/test split
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size= 0.25)

################ find word count and store data as matrix
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
x_train_count.toarray()

################ train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

################ pre-test ham
email_ham = ["hello there, today we have a meeting together."]
email_ham_count = cv.transform(email_ham)
result_predict_1 = model.predict(email_ham_count)
print("result ham check: ",result_predict_1)

################ pre-test spam
email_spam = ["click for reward money 1000$."]
email_spam_count = cv.transform(email_spam)
result_predict_2 = model.predict(email_spam_count)
print("result spam check: ", result_predict_2)

################ test model
x_test_count = cv.transform(x_test)
result_score = model.score(x_test_count, y_test)
print(result_score)



