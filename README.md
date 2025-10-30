
# 📧 Spam Mail Detection 

This project is a **Spam Mail Detection System** built using **Python** and **Machine Learning**.  
It classifies email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and supervised learning algorithms.

---

## 🚀 Project Structure

```

spam_detector/
│
├── spam_detector.py       # Main script with training, visualization, and model evaluation
├── requirements.txt       # All dependencies required for the project
├── README.md              # Project documentation

````

---

## 🧠 Objective

To develop a model that automatically detects whether an email is **spam** or **not spam**, helping filter unwanted or fraudulent messages effectively.

---

## 📂 Dataset Information

The dataset used for this project contains SMS messages labeled as **spam** or **ham**.

**Dataset Link:** [Download CSV from Google Drive](https://drive.google.com/uc?id=1CsT32kpqY1cnz4h38OzDtBmr6-avTTiH)

| Column Name | Description |
|--------------|--------------|
| `v1` | Label (`spam` / `ham`) |
| `v2` | The text message content |

---

## ⚙️ Steps Involved

### 1️⃣ Import Libraries & Load Dataset
Load and inspect the dataset.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
````

---

### 2️⃣ Data Cleaning & Preprocessing

* Convert text to lowercase
* Remove punctuation and stopwords
* Tokenize messages
* Label encode spam/ham

---

### 3️⃣ Data Visualization

Visualizations provide insights into dataset distribution and characteristics.

```python
sns.countplot(data=df, x='label')
plt.title("Spam vs Ham Message Distribution")
plt.show()

df['length'] = df['message'].apply(len)
sns.histplot(df[df['label']=='ham']['length'], color='green', label='Ham', bins=50)
sns.histplot(df[df['label']=='spam']['length'], color='red', label='Spam', bins=50)
plt.legend()
plt.title("Message Length Distribution")
plt.show()
```

---

## 🧩 Model Building Process

### ➤ Step 1: Vectorization

Text is converted into numerical form using **CountVectorizer**.

### ➤ Step 2: Model Training

Model used: **Multinomial Naive Bayes**

### ➤ Step 3: Model Evaluation

The model is evaluated using accuracy, precision, recall, and F1-score.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
model = MultinomialNB()
model.fit(x_train_cv, y_train)
predictions = model.predict(cv.transform(x_test))
```

---

## 📊 Results & Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~98%  |
| Precision | High  |
| Recall    | High  |

---

## 🖼️ Screenshots

### 📈 Spam vs Ham Distribution

<img src="https://drive.google.com/uc?export=view&id=1XE8zFob0oQyFeZOzEwbd9c3XoqzjdndZ" width="700"/>

### 🧠 Model Output Example

<img src="https://drive.google.com/uc?export=view&id=14ajiBH9Ar9bWV7_A4NiqUOF-iUfd14yp" width="700"/>

### 📉 Confusion Matrix Visualization

<img src="https://drive.google.com/uc?export=view&id=1ZvBSNvXLGcD7p_6lLzYRldB8VhGJiRzO" width="700"/>

---

## 🧰 Requirements

Create a `requirements.txt` file with the following dependencies:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
```

Install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

1. Clone the repository or download the project folder.
2. Install dependencies using `requirements.txt`.
3. Run the Python script:

```bash
python spam_detector.py
```

4. The system will display metrics, confusion matrix, and visualizations.

---

## 👩‍💻 Author

**Komal Kale**
🎓 AI & Data Science Enthusiast
🔗 [GitHub Profile](https://github.com/Komalkale2)
🔗 [LinkedIn](https://www.linkedin.com/in/komal-kale-8abb902b4/)

---

## ⭐ Future Scope

* Deployment as a **Flask / Streamlit Web App**
* Integration with **live email filtering APIs**
* Experiment with **TF-IDF** and **Word Embeddings**

---

## 🏁 Conclusion

This project demonstrates how **Machine Learning and NLP** techniques can effectively classify email messages as **Spam or Ham** with high accuracy, providing a foundation for real-world spam filtering systems.

```

---

Would you like me to add a short **"Demo Results" section** (e.g., sample predictions of spam and ham messages) before the screenshots? It makes the README look even more professional.
```
