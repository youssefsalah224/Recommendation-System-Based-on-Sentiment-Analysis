# Sentiment‑Aware Recommendation System for Egyptian Arabic Reviews

**Python • NLP • Sentiment Analysis • Hybrid Recommendation Engine**

A hybrid recommendation system that integrates sentiment analysis on Egyptian Arabic reviews (collected from Twitter and Google Play) to enhance personalization, context-awareness, and predictive accuracy.

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Project Flow](#project-flow)
- [Feature Extraction](#feature-extraction)
- [Models Used](#models-used)
- [Technologies & Tools](#technologies--tools)
- [Installation & Usage](#installation--usage)
- [Results & Evaluation](#results--evaluation)
- [Future Enhancements](#future-enhancements)


---

## Overview
This project implements a sentiment-aware recommendation system tailored for Egyptian Arabic language reviews. It bridges the gap between traditional collaborative filtering techniques and the subjective, emotional dimension of user feedback.

---

## Problem Statement
- Most recommendation systems rely solely on structured numeric ratings or metadata.
- However, valuable user sentiment exists in unstructured Arabic text, especially on platforms lacking proper rating systems.
- By leveraging sentiment analysis, the system can extract emotional context to make smarter, more personal recommendations.

---

## Architecture
The system follows a modular pipeline:
1. **Text Preprocessing** – Custom cleaning for Arabic (emoji removal, diacritics, stopwords).
2. **Vectorization** – Transform Arabic text into numerical format using TF-IDF and Average Word2Vec.
3. **Sentiment Modeling** – Train sentiment classifiers using traditional ML, deep learning, and sequential models.
4. **Recommendation Engine** – Combine sentiment output with filtering mechanisms to drive personalized suggestions.

---

## Project Flow
1. **Data Collection**
   - Arabic reviews scraped from Twitter and Google Play focusing on Egyptian companies.

2. **Data Cleaning**
   - Headline emojis removal
   - Diacritics removal
   - Arabic stopwords removal

3. **Feature Engineering**
   - Tokenization and vectorization using:
     - TF-IDF
     - Average Word2Vec embeddings

4. **Modeling**
   - Apply various models for sentiment classification (details below).

5. **Evaluation**
   - Metrics such as Accuracy, Precision, Recall, and F1-score used to assess sentiment classification and recommendation quality.

6. **Prediction**
   - Test predictions on unseen samples to evaluate system robustness and recommendation validity.

---

## Feature Extraction
We convert textual Arabic data into numerical form using two techniques:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Captures the importance of terms in each review.
- **Average Word2Vec Embedding**: Computes the average vector of all word embeddings in a review, capturing semantic meaning.

---

## Models Used

### Traditional Machine Learning Models
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes (NB)
- Linear SVM (LSVM)

### Deep Learning Models
- Forward Neural Network (FNN)
- Convolutional Neural Network (CNN)

### Sequential Models
- Recurrent Neural Network (RNN)
- Gated Recurrent Unit (GRU)
- Long Short-Term Memory (LSTM)

> ✅ **Best Performing Setup**:
> - **Text Representation**: Average Word2Vec  
> - **Model**: GRU (Gated Recurrent Unit)

---

## Technologies & Tools
- **Languages**: Python
- **Libraries**:
  - NLP: NLTK, spaCy, re
  - ML/DL: scikit-learn, TensorFlow/Keras
  - Embeddings: gensim (Word2Vec)
- **Data Handling**: pandas, NumPy
- **IDE**: Jupyter Notebook, VSCode

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/youssefsalah224/Recommendation-System-Based-on-Sentiment-Analysis.git
cd Recommendation-System-Based-on-Sentiment-Analysis

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the project
jupyter notebook
# or execute scripts as needed
