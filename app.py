import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dataset_path = r"C:\Users\nivet\Downloads\ml\project\WELFake_Dataset.csv"  
data = pd.read_csv(dataset_path)


print(data.head())
print(data.info())


data = data.dropna(subset=['text'])



# Splitting the dataset
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test the model with a new input
def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "REAL" if prediction[0] == 1 else "FAKE"

# Example usage
sample_news = "The government has announced a new tax reform plan."
print(f"The news article is predicted as: {predict_news(sample_news)}")
