# A Project on abusive language classification # 
# PrimeLine – Tamil Abusive Comment Detection

## Overview

PrimeLine is a **Natural Language Processing (NLP)** project designed to detect **abusive or offensive comments written in Tamil**.

Social media platforms receive a large number of comments every day, and some may contain harmful language. Manually filtering such comments can be difficult. This project aims to automatically identify abusive Tamil comments using **machine learning techniques**.

The system analyzes Tamil text and predicts whether the comment is **abusive** or **non-abusive**, helping improve online moderation.

## Features

* Tamil text processing
* Abusive comment detection
* Machine learning based classification
* Data preprocessing pipeline
* Model training and evaluation

## Project Structure
primeline_tamil_abusive
│
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
│
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation



## Installation

1. Clone the repository


git clone https://github.com/Nithya-svg/primeline_tamil_abusive.git
cd primeline_tamil_abusive


2. Install the required dependencies
   
pip install -r requirements.txt


## Usage

Run the training script:

python train.py

Example prediction:

python

from model import predict

text = "உன் கருத்து மிகவும் மோசமானது"
result = predict(text)
print(result)

Example Output
Abusive


## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Natural Language Processing (NLP)

## Applications

* Social media moderation
* Hate speech detection
* Comment filtering systems
* Online community monitoring

## Future Improvements

* Use deep learning models (LSTM / Transformers)
* Expand the Tamil abusive dataset
* Build a web interface for real-time detection
* Support multiple languages


## License

This project is licensed under the **MIT License**.

