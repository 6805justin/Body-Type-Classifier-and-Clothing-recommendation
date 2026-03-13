# Body-Type-Classifier-and-Clothing-recommendation

Overview

This project builds an AI-powered web application that predicts a user's body type from an uploaded
image and recommends outfits based on body shape, occasion, and weather conditions. The system
combines computer vision and deep learning to deliver personalized fashion suggestions.

Problem Statement

Choosing outfits that suit body shape and context (event, weather) can be difficult. This project uses a
Convolutional Neural Network (CNN) to automatically classify body types and generate fashion
recommendations.

Features

• Upload image for body type prediction

• CNN-based image classification

• 5 body type categories: Hourglass, Pear, Apple, Rectangle, Inverted Triangle

• Outfit recommendation based on body type, occasion, and weather

Model Architecture

The project uses a Convolutional Neural Network (CNN) trained on categorized body shape images.


Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Output Classes: 5

Model Accuracy: ~89%

Tech Stack

Python, TensorFlow / Keras, Flask, OpenCV, HTML / CSS

Future Improvements

• Expand dataset for better generalization

• Add deep learning-based recommendation engine

• Deploy the model on cloud infrastructure

• Integrate real-time camera capture

Project Goal

Demonstrate how machine learning and computer vision can be integrated into an end-to-end
application for personalized fashion recommendations.
