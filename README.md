# Sentiment-Analysis-of-Customer-Reviews
# Download And Extract DataSet
!wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xzf aclImdb_v1.tar.gz
# Load Into a DataFrame
import os
import pandas as pd

def load_imdb(path, dataset_type):
    texts, labels = [], []
    base = os.path.join(path, dataset_type)
    for sentiment in ["pos", "neg"]:
        folder = os.path.join(base, sentiment)
        for fname in os.listdir(folder):
            with open(os.path.join(folder, fname), encoding='utf8') as f:
                texts.append(f.read())
                labels.append(1 if sentiment == "pos" else 0)
    return pd.DataFrame({"review": texts, "label": labels})

train_df = load_imdb("aclImdb", "train")
test_df = load_imdb("aclImdb", "test")
print(train_df.shape, test_df.shape)
train_df.head()
# Preprocess the Text Data
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply to both train and test
train_df['clean_review'] = train_df['review'].apply(preprocess)
test_df['clean_review'] = test_df['review'].apply(preprocess)

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_review'])
X_test = vectorizer.transform(test_df['clean_review'])

y_train = train_df['label']
y_test = test_df['label']
# Train a Machine Learning Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Predict on New Review
def predict_sentiment(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "🟢 Positive" if pred == 1 else "🔴 Negative"


print(predict_sentiment("This movie was fantastic! Really enjoyed it."))
print(predict_sentiment("Worst movie ever. Total waste of time."))
