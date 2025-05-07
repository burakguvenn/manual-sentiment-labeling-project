# Product Review Sentiment Analysis

This repository contains a Natural Language Processing (NLP) project for binary sentiment analysis (Positive/Negative) on product reviews.

## Project Objective

The primary goal is to develop a machine learning model that automatically classifies the sentiment of text-based product reviews. The specific objectives are:

- To experience the process of manually creating a labeled dataset for supervised learning.
- To implement an end-to-end NLP pipeline, including data preprocessing, feature extraction, modeling, and evaluation, on a limited and class-imbalanced dataset.
- To analyze model performance using standard metrics and interpret the impact of data limitations and class imbalance.

## Dataset

The dataset is sourced from Amazon product reviews available on Kaggle: [Amazon Review Dataset](https://www.kaggle.com/datasets/mehmetisik/amazon-review/data). A sample from this raw dataset was manually labeled with sentiment classes (Positive/Negative) for this project.

**Note**: The complete manually labeled dataset is not included in this repository. To reproduce the project, obtain the raw data from the specified source and perform labeling as described in the project report.

## Requirements

To run the project, ensure you have:

- **Python 3** installed.
- The following Python libraries:
  - `pandas`
  - `scikit-learn`
  - `nltk` (included in Python's standard library)

Install the required libraries using:


pip install pandas scikit-learn nltk

**Note**: Some NLTK components (e.g., `stopwords`, `punkt`) may download automatically on first run. If issues occur, uncomment the relevant lines in `process_data.py` to manually download them.

## Usage

Follow these steps to run the project:

1. Install the required libraries (see "Requirements").
2. Obtain the raw data from the Kaggle source and manually label a sample as per the project methodology. Save the labeled data as `yorumlar_etiketlenecek.csv` in the project's main directory.
3. Open a Terminal or Command Prompt and navigate to the project directory.
4. Run the main script:

```bash
python main.py
