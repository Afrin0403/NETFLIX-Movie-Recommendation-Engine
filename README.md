# NETFLIX-Movie-Recommendation-Engine
# Movie Recommendation Engine - Data Science Project

Welcome to the Movie Recommendation Engine! This repository contains the code and resources for building a personalized movie recommendation system using data science techniques. The project leverages collaborative filtering, content-based filtering, and hybrid methods to suggest movies based on user preferences.

## Project Overview

The goal of this project is to create an intelligent movie recommendation engine that can provide movie suggestions based on individual preferences. By analyzing user ratings, movie features, and historical data, the engine predicts which movies a user is likely to enjoy. 

### Key Features:
- **Collaborative Filtering**: Recommends movies based on similar user preferences.
- **Content-Based Filtering**: Suggests movies similar to those a user has already watched or liked based on movie features like genres, directors, and actors.
- **Hybrid Approach**: Combines both collaborative and content-based techniques to improve recommendation accuracy.
- **Evaluation Metrics**: Evaluates the recommendation engine performance using metrics like RMSE, Precision, Recall, and F1-Score.

## Technologies Used

- **Python** (for data processing and modeling)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical calculations)
- **Scikit-learn** (for machine learning models)
- **Surprise** (for collaborative filtering models)
- **Matplotlib & Seaborn** (for data visualization)
- **Google Collab** (for development and documentation)

## Dataset

This project uses the [kaggle dataset](https://www.kaggle.com/code/laowingkin/netflix-movie-recommendation/input), which contains user ratings and movie metadata. The dataset is widely used for movie recommendation system projects and is available in various sizes.


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Afrin0403/NETFLIX-Movie-Recommendation-Engine/
2. Install the dependencies
To install all necessary libraries, you can use pip:

bash
pip install -r requirements.txt
This will install all the required packages such as Pandas, NumPy, Scikit-learn, Surprise, Matplotlib, etc.

Usage
Data Preprocessing: Prepare the dataset by cleaning and normalizing the data, handling missing values, and splitting it into training and test sets.

Collaborative Filtering:

Use the Surprise library to implement user-based or item-based collaborative filtering.
Generate movie recommendations based on user-item interaction data.
Content-Based Filtering:

Analyze movie features (like genre, director, and cast) to recommend similar movies to the user.
Hybrid Approach: Combine collaborative and content-based filtering for better results.

Evaluation:

Evaluate the model using metrics such as RMSE (Root Mean Squared Error), Precision, Recall, and F1-Score.
Visualize model performance using Matplotlib and Seaborn.
Example:

# Import necessary libraries
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the data
reader = Reader()
data=Dataset.load_from_df(netflix_dataset[:100000], reader)
# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD algorithm
svd = SVD()
algo.fit(trainset)

# Make predictions
predictions = algo.test(testset)

# Evaluate the performance
accuracy.rmse(predictions)
Results
The system provides accurate movie recommendations based on user preferences. Below are the metrics used to evaluate the system:

RMSE (Root Mean Squared Error)
Precision, Recall, Estimate score
These metrics are essential for understanding the recommendation engine's performance and ensuring its effectiveness in providing relevant movie suggestions.



