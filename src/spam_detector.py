"""
📧 EMAIL SPAM DETECTION
--------------------------------
Detects whether an email/SMS message is Spam or Not Spam using
NLP (TF-IDF Vectorization) and a Naive Bayes Classifier.

Dataset: UCI SMS Spam Collection (via Google Drive)
"""

# ===============================================
# STEP 1: Import Libraries
# ===============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ===============================================
# STEP 2: Load Dataset
# ===============================================
print("📦 Loading dataset...")

file_id = "1CsT32kpqY1cnz4h38OzDtBmr6-avTTiH"
download_url = f"https://drive.google.com/uc?id={file_id}"

df = pd.read_csv(download_url, encoding='latin-1')

# Clean columns
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
else:
    df.columns = ['label', 'message']

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("✅ Dataset loaded and cleaned successfully!")
print(df.head())

# ===============================================
# STEP 3: Data Visualization
# ===============================================
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df, palette='Set2')
plt.title("Spam vs Ham Message Distribution")
plt.savefig("distribution.png")
plt.close()

# ===============================================
# STEP 4: TF-IDF Vectorization
# ===============================================
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================================
# STEP 5: Model Training
# ===============================================
model = MultinomialNB()
model.fit(X_train, y_train)

# ===============================================
# STEP 6: Model Evaluation
# ===============================================
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)

print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy: {accuracy}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# ===============================================
# STEP 7: Save Model
# ===============================================
joblib.dump(model, "spam_detector_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully!")

# ===============================================
# STEP 8: Custom Message Test
# ===============================================
def check_spam(message):
    msg_vec = vectorizer.transform([message])
    pred = model.predict(msg_vec)[0]
    return "🚨 Spam" if pred == 1 else "✅ Not Spam"

# Test examples
examples = [
    "Congratulations! You have won a $500 gift card.",
    "Hey, are we meeting for lunch tomorrow?",
    "URGENT: Your bank account has been suspended. Verify now!"
]

print("\n🔍 SAMPLE TESTS")
for msg in examples:
    print(f"\nMessage: {msg}\nPrediction: {check_spam(msg)}")

print("\n✅ Spam detection complete!")

