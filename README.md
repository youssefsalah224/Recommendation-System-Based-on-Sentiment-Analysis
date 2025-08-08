# 🇪🇬 Sentiment-Aware Recommendation System on Egyptian Arabic Reviews

**Arabic NLP • Sentiment Analysis • Machine Learning • Deep Learning**

A recommendation system powered by sentiment analysis on Egyptian Arabic reviews collected from Twitter and Google Play. This project combines traditional and deep learning models with word embeddings to extract user sentiment and enhance personalized recommendations.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Feature Extraction](#feature-extraction)
- [Modeling](#modeling)
  - [Traditional ML Models](#traditional-ml-models)
  - [Deep Learning Models](#deep-learning-models)
  - [Sequential Models](#sequential-models)
- [Best Results](#best-results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [License](#license)

---

## 🧠 Overview

This project builds a hybrid **sentiment-based recommendation system** for Egyptian companies, using **natural language processing (NLP)** on Arabic user reviews. We preprocess the text, convert it to numerical vectors, classify its sentiment using various models, and apply these insights in a recommendation pipeline.

---

## 🗃️ Dataset

- **Sources**:  
  - Twitter  
  - Google Play  
- **Language**: Egyptian Arabic  
- **Domain**: Reviews about Egyptian companies  
- The dataset was manually labeled and cleaned to be ready for NLP-based sentiment analysis.

---

## 🧹 Data Cleaning

Preprocessing is essential when dealing with Arabic text, especially dialects like Egyptian Arabic.

```text
1. Remove headline emojis
2. Remove Arabic diacritics (e.g., َ ً ُ ٌ etc.)
3. Remove Arabic stop words (common but non-informative words)
