import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_json("News_Category_Dataset_v3.json",lines=True)

# Features and labels
x = df["headline"]
y = df["category"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert text into numerical vectors
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

# Prediction
y_pred = model.predict(x_test_tfidf)

# Accuracy
score = accuracy_score(y_test, y_pred)
print("Accuracy:", round(score * 100, 2), "%")

# Test your own news
news = ["India won the T20 World Cup in 2024."]
news_vector = tfidf.transform(news)

prediction = model.predict(news_vector)

if prediction[0] == "REAL":
    print("Real News")
else:
    print("Fake News")