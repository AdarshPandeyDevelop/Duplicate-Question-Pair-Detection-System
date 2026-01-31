# Duplicate-Question-Pair-Detection-System
Duplicate Question Pair Detection System
🔁 Duplicate Question Pair Detection System
📌 Project Overview

This project implements a machine learning–based system to detect duplicate question pairs using Natural Language Processing (NLP) techniques and similarity-based feature engineering.

The model determines whether two given questions are identical or highly similar in structure and wording, rather than understanding deep semantic meaning.
It is inspired by problems such as the Quora Question Pairs dataset.

A Streamlit web application is provided for real-time interaction and inference.

🎯 Objective

To classify whether two questions are:

✅ Duplicate

❌ Not Duplicate

based on text similarity, token overlap, and string-based comparison, not semantic embeddings.

⚠️ Important Note (Scope & Limitation)

⚠️ This system does NOT perform semantic similarity detection (e.g., meaning-level understanding using BERT or embeddings).

It focuses on:

Identical or near-identical wording

Structural similarity

Lexical overlap

String and token-level matching

This makes it a classical NLP + Machine Learning solution, not a deep learning–based semantic model.

🧠 System Architecture

User Input (Question 1 & Question 2)
        ↓
Text Preprocessing
        ↓
Feature Engineering
        ↓
Bag of Words Vectorization
        ↓
Trained ML Model
        ↓
Duplicate / Not Duplicate

🛠️ Technologies Used

Python

Streamlit – Web interface

Scikit-learn – ML model & vectorization

NLTK / Custom Stopwords

FuzzyWuzzy – String similarity

BeautifulSoup – HTML cleaning

NumPy

Pickle – Model persistence

🧹 Text Preprocessing

Each question undergoes extensive preprocessing, including:

Lowercasing

Special character normalization (₹, $, %, etc.)

Number normalization (1,000 → 1k, 1,000,000 → 1m)

Decontraction (can't → can not)

HTML tag removal

Punctuation removal

Whitespace normalization

This ensures consistent and clean textual input for feature extraction.

🧩 Feature Engineering

The model uses a hybrid feature approach combining hand-crafted similarity features with Bag-of-Words vectors.

🔢 Basic Features

Question lengths

Word counts

Common word counts

Ratio of common to total words

🧠 Token-Based Features

Common non-stopword ratios

Common stopword ratios

Token overlap ratios

First word match

Last word match

📏 Length-Based Features

Absolute length difference

Average question length

Longest common substring ratio

🔍 Fuzzy Matching Features

Fuzzy ratio

Partial ratio

Token sort ratio

Token set ratio

🧾 Bag of Words (BoW)

CountVectorizer applied separately to both questions

Combined with handcrafted features into a single feature vector

🤖 Machine Learning Model

Supervised classification model

Trained on labeled question pairs

Outputs:

1 → Duplicate

0 → Not Duplicate

The trained model and vectorizer are stored using pickle for inference.

🌐 Web Application (Streamlit)

The Streamlit app allows users to:

Enter two questions

Click Find

Instantly receive a prediction:

Duplicate

Not Duplicate

This makes the model easy to test and demonstrate.

▶️ How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/duplicate-question-detection.git
cd duplicate-question-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

📈 Use Cases

Detecting duplicate questions in Q&A platforms

Data cleaning for NLP datasets

Avoiding redundant user queries

Demonstrating classical NLP feature engineering

🚀 Future Improvements

Add semantic embeddings (BERT / SBERT)

Replace BoW with TF-IDF or Word Embeddings

Optimize fuzzy matching for scalability

Improve generalization to paraphrased questions

🏁 Conclusion

This project demonstrates a classical NLP-based machine learning pipeline for duplicate question detection using text preprocessing and similarity features.
It is ideal for understanding feature engineering, string similarity, and ML deployment, while clearly defining its semantic limitations.
