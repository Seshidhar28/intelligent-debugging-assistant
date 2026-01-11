# Intelligent Debugging Assistant

An ML-based debugging assistant that analyzes backend logs, clusters similar errors using NLP techniques, and identifies the most probable root causes.

## Features
- Parses raw application logs into structured data
- Uses TF-IDF vectorization for error message representation
- Applies KMeans clustering to group similar failures
- Ranks services and functions by failure frequency

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NLP (TF-IDF)
- Unsupervised Machine Learning

## How to Run
```bash
pip install pandas scikit-learn
python debugging_assistant.py
