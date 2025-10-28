🔥 Prali Fire (Stubble Burning) Prediction Model

This project is a machine learning web application built with Streamlit that predicts the likelihood of a satellite-detected fire being a "Prali fire" (stubble burning).

The model is a Random Forest Classifier trained on the physical characteristics of the fire, such as brightness, temperature, and energy output, intentionally ignoring location and date data to build a more generalized model.

🚀 Project Overview

The project consists of three main components:

    Data Splitting: A script to split the original final_data (1).csv into 80% training and 20% testing sets.

    Model Training (create_model.py): A script that loads training_data.csv, trains a "balanced" Random Forest model on the fire's physical properties, and saves the entire model pipeline (preprocessors + model) as prali_fire_model.pkl.

    Web Application (app.py): A Streamlit app that loads the .pkl file, provides a user-friendly form, and serves real-time predictions with confidence scores.

📂 Project Structure

For the application to run correctly, your folder must be organized as follows:

your-project-folder/
├── final_data (1).csv       # Your original, complete dataset
├── training_data.csv      # (Will be created by split_data.py)
├── testing_data.csv       # (Will be created by split_data.py)
├── create_model.py        # Script to train and save the model
├── app.py                 # The Streamlit web application
├── prali_fire_model.pkl   # (Will be created by create_model.py)
└── README.md              # This file

🛠️ Setup and Installation

Follow these steps to run the project on your local machine.

1. Prerequisites

You must have Python 3.7+ installed.

2. Install Required Libraries

Open your terminal and install all necessary Python libraries: