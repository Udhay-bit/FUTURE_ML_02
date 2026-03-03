Support Ticket Classification & Prioritization System
Project Overview
This project aims to develop a Machine Learning system capable of automatically classifying customer support tickets into predefined categories (e.g., Billing Inquiry, Technical Issue) and assigning a priority level (High, Medium, Low). The ultimate goal is to streamline customer support operations by enabling faster routing of tickets, improving response times for urgent issues, and optimizing resource allocation. This system moves beyond traditional manual sorting to provide an intelligent decision-support tool for businesses.

Objective
The primary objective was to build an ML system that could:

Read and understand the textual content of customer support tickets.
Automatically classify tickets into relevant categories.
Predict the urgency or priority level of each ticket.
This automation is crucial for enhancing customer satisfaction, reducing operational backlogs, and freeing up support agents to focus on complex problem-solving.

Dataset
Dataset Used: Customer Support Ticket Dataset (from Kaggle)

Source: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
Description: This dataset contains structured support ticket data, including textual descriptions of issues, ticket types, and priority levels, making it ideal for text classification and prioritization tasks.
Methodology
The project followed a comprehensive machine learning pipeline:

1. Data Loading and Exploration
The dataset was loaded into a pandas DataFrame.
Initial exploratory data analysis (EDA) was performed to understand data structure, identify missing values, and analyze distributions of key categorical variables ('Ticket Type', 'Ticket Priority', 'Customer Gender', 'Ticket Status', 'Ticket Channel').
2. Text Preprocessing
Customer support ticket text ('Ticket Subject' and 'Ticket Description') was consolidated into a single consolidated_text feature and meticulously cleaned through the following steps:

Lowercasing: Converting all text to lowercase for consistency.
Punctuation and Number Removal: Eliminating non-alphabetic characters to reduce noise.
Stopword Removal: Removing common English stopwords (e.g., 'the', 'is', 'a') that carry little semantic value.
Lemmatization: Reducing words to their base or dictionary form to normalize vocabulary variations.
3. Feature Extraction
Two different techniques were explored to convert the cleaned text into numerical features:

TF-IDF (Term Frequency-Inverse Document Frequency): A statistical measure reflecting how important a word is to a document in a collection or corpus. This resulted in a sparse matrix of (8469, 5000) dimensions.
Word Embeddings (Word2Vec): Pre-trained Word2Vec embeddings (word2vec-google-news-300) were used to capture semantic relationships between words. Document embeddings were created by averaging the word vectors for each word in a ticket, resulting in a dense feature vector of size (8469, 300).
4. Model Training
Two classification tasks were addressed: 'Ticket Type' classification and 'Ticket Priority' prediction. For each task, the data was split into 80% training and 20% testing sets. Two types of models were trained:

Logistic Regression: Used as a baseline model with TF-IDF features.
Support Vector Machine (SVC): Employed with Word Embeddings features for a more advanced approach.
5. Model Evaluation
Both sets of models were evaluated using standard classification metrics:

Classification Report: Providing precision, recall, F1-score, and support for each class.
Confusion Matrix: Visualizing the true vs. predicted labels to understand misclassifications.
Results and Performance
Unfortunately, both the initial and advanced models demonstrated poor performance, struggling to classify ticket types and priorities significantly better than random chance.

Logistic Regression (with TF-IDF):

'Ticket Type' Accuracy: Approximately 0.20 (out of 5 classes, random guess would be 0.20).
'Ticket Priority' Accuracy: Approximately 0.26 (out of 4 classes, random guess would be 0.25).
Low precision, recall, and F1-scores across all categories and priority levels, indicating frequent misclassifications.
Support Vector Machine (with Word Embeddings):

'Ticket Type' Accuracy: Approximately 0.19 (marginally lower than Logistic Regression).
'Ticket Priority' Accuracy: Approximately 0.25 (comparable to random guessing).
Similar low performance metrics to the Logistic Regression models, showing no substantial improvement despite using more sophisticated feature extraction and modeling techniques.
These results indicate that the models were not effective in capturing the intricate patterns and semantic nuances required for accurate classification and prioritization of support tickets.

Potential Reasons for Suboptimal Performance
Complexity of Text Data: The subtle linguistic cues and context within support tickets might be too complex for current feature representations and linear models to capture.
Limitations of Averaging Word Embeddings: Simply averaging Word2Vec vectors may lose crucial information about word order, grammar, and sentence structure.
General Pre-trained Embeddings: The 'word2vec-google-news-300' model is general-purpose and might not adequately represent the specific jargon and context of customer support language.
Model Simplicity: Linear models like Logistic Regression and linear-kernel SVC might be too simplistic for the highly non-linear relationships in text data.
Placeholder Impact: Generic placeholders like {product_purchased} in the original text could mask important product-specific information vital for classification.
Future Work and Improvements
To build a robust and effective support ticket classification and prioritization system, the following next steps are recommended:

Explore Advanced Text Representations: Implement and evaluate contextual embeddings like BERT, RoBERTa, or GPT. These models are designed to understand word context within sentences more effectively.
Utilize Deep Learning Architectures: Transition to more powerful models such as Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), or Convolutional Neural Networks (CNNs), which are better suited for processing sequential text data.
Domain-Specific Embeddings: Consider training custom Word2Vec or other embeddings on a larger corpus of customer support data to better capture domain-specific terminology and relationships.
Sophisticated Feature Engineering: Investigate methods to extract or interpret information from placeholders like {product_purchased}. This could involve external data sources or more advanced NLP techniques.
Hyperparameter Tuning & Ensemble Methods: Systematically tune model hyperparameters and explore ensemble techniques to combine the strengths of multiple models.
Data Augmentation: If feasible, augment the training data with synthetically generated tickets or external relevant data to improve model generalization.
By iteratively exploring these advanced techniques, we aim to develop a machine learning system that can significantly enhance the efficiency and effectiveness of customer support operations.
