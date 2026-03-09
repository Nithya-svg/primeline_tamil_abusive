# A Project on abusive language classification #
PrimeLine – Tamil Abusive Comment Detection
Introduction

PrimeLine is a project that focuses on identifying abusive or offensive comments written in Tamil. Social media platforms today receive a large number of comments every day, and many of them may contain harmful or abusive language. Manually filtering such content can be difficult and time-consuming.

This project uses Natural Language Processing (NLP) and machine learning techniques to automatically detect whether a Tamil comment is abusive or not. The goal is to help create safer and more respectful online spaces.

Objective

The main objective of this project is to build a model that can:

Analyze Tamil text comments

Identify whether a comment is abusive or non-abusive

Help automate the moderation of online discussions

Features

Detects abusive Tamil comments

Processes Tamil language text

Uses machine learning for classification

Includes data preprocessing and model training steps

Can be extended for real-world content moderation systems

Project Structure
primeline_tamil_abusive
│
├── data/              # Dataset used for training and testing
├── notebooks/         # Jupyter notebooks for experiments
├── src/               # Source code files
├── requirements.txt   # Required Python libraries
└── README.md          # Project documentation
Dataset

The dataset used in this project contains Tamil comments labeled based on their nature, such as:

Abusive

Non-abusive

Offensive (if applicable)

These labeled comments help the model learn how to distinguish harmful content from normal text.

Methodology

The project follows these steps:

1. Data Preprocessing

Before training the model, the text is cleaned by:

Removing unnecessary characters

Converting text into a suitable format

Tokenizing the words

2. Feature Extraction

Techniques like TF-IDF are used to convert Tamil text into numerical form so that machine learning models can understand it.

3. Model Training

A machine learning algorithm is trained using the processed dataset to classify comments.

4. Evaluation

The performance of the model is checked using metrics such as:

Accuracy

Precision

Recall

F1 Score

Installation

Clone the repository:

git clone https://github.com/Nithya-svg/primeline_tamil_abusive.git
cd primeline_tamil_abusive

Install the required libraries:

pip install -r requirements.txt
Usage

Run the training script:

python train.py

You can also test the model with a sample Tamil comment to check whether it is abusive or not.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Natural Language Processing (NLP)

Applications

This project can be useful for:

Social media comment moderation

Hate speech detection

Online community monitoring

AI-based content filtering systems

Future Improvements

Possible improvements for this project include:

Using deep learning models such as LSTM or Transformers

Expanding the dataset with more Tamil comments

Building a web interface for real-time comment detection

Supporting multiple languages

License

This project is available under the MIT License.
