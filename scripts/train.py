import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from process import process_dataframe
import joblib


df = pd.read_csv("data/tweets.csv", encoding='latin-1', header=None)
df.columns = ['sentiments','tweet_id','date','query','user','text']
df = df[['sentiments','text']]
df['sentiments'] = df['sentiments'].replace(4, 1)  


df = process_dataframe(df)


vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 3))
x = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiments']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=5000, n_jobs=-1, solver="saga")
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


print("Accuracy is:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, "outputs/model.pkl")
joblib.dump(vectorizer, "outputs/vectorizer.pkl")