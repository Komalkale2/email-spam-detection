
# 📧 Email Spam Detection 

This project aims to detect whether a given **email or SMS message** is *Spam* or *Not Spam* using **Natural Language Processing (NLP)** and **Machine Learning**.  
It uses **TF-IDF vectorization** with a **Naive Bayes classifier** to analyze text and accurately classify messages.

---

## 🧠 Overview

Spam detection is one of the most common text classification problems.  
In this project, we build a model that automatically identifies spam messages using text processing, feature extraction, and machine learning.

The goal is to:
- Clean and preprocess text messages  
- Extract features using **TF-IDF (Term Frequency–Inverse Document Frequency)**  
- Train a **Naive Bayes** model  
- Predict and evaluate the accuracy on unseen messages  

---

## 📂 Dataset

- **Dataset Name:** SMS Spam Collection Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- **Alternate Download Link:** [Google Drive](https://drive.google.com/file/d/1CsT32kpqY1cnz4h38OzDtBmr6-avTTiH/view?usp=sharing)  

**Dataset Details:**
- Total Messages: 5,574  
- Columns:
  - `label` — "spam" or "ham" (non-spam)  
  - `message` — the SMS or email text  

---

## ⚙️ Technologies Used

- **Programming Language:** Python  
- **Libraries & Tools:**
  - pandas  
  - numpy  
  - scikit-learn  
  - nltk  
  - matplotlib  
  - wordcloud  

---

## 🧩 Project Structure

```

email-spam-detection/
│
├── dataset/
│   └── spam.csv
│
├── notebook/
│   └── email_spam_detection.ipynb
│
├── src/
│   └── spam_detector.py
│
├── requirements.txt
└── README.md

````

---

## 🚀 Installation & Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Komalkale2/email-spam-detection.git
cd email-spam-detection
````

### 2️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Project

You can run the Jupyter Notebook or Colab file:

```bash
jupyter notebook notebook/email_spam_detection.ipynb
```

Or open it directly in **Google Colab**.

---

## 🧹 Data Preprocessing Steps

1. **Load Data** – Read CSV dataset using pandas
2. **Clean Text** – Convert to lowercase, remove stopwords, punctuation, and extra spaces
3. **Tokenize** – Split text into meaningful words
4. **Vectorize** – Convert text to numerical features using TF-IDF
5. **Train Model** – Train with Multinomial Naive Bayes
6. **Evaluate** – Measure accuracy, precision, recall, and F1-score

---

## 🧪 Example Test

```python
test_messages = [
    "Congratulations! You have won a free vacation to Maldives!",
    "Let's have lunch tomorrow at 1?"
]
predictions = model.predict(vectorizer.transform(test_messages))
print(predictions)
# Output: ['spam', 'ham']
```

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97%   |
| Precision | 96%   |
| Recall    | 95%   |
| F1-Score  | 95%   |

---

## 🎯 Future Improvements

* Deploy the model using **Flask** or **Streamlit**
* Add **real email header analysis**
* Implement **BERT-based** transformer models
* Enable **real-time email classification**

---

* WordCloud of Spam vs. Ham messages
* Confusion Matrix
* Model Accuracy Graph

---

## 👩‍💻 Author

**Komal Kale**
📍 Data Science & Machine Learning Enthusiast

* 🔗 [GitHub Profile](https://github.com/Komalkale2)
* 💼 [LinkedIn](https://www.linkedin.com/in/komal-kale-8abb902b4/)

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with proper credit.

---

### ⭐ If you like this project, don’t forget to star the repository!


