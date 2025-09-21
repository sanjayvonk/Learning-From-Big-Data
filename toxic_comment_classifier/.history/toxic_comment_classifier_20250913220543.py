import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
from nltk.stem import WordNetLemmatizer

# Load the training data
train_df = pd.read_csv('data/train.csv')

# Split the data into training and validation sets
train_text, val_text, train_toxic, val_toxic = train_test_split(
    train_df['comment_text'],
    train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
    test_size=0.2,
    random_state=42
)

# Preprocessing: remove punctuation, convert to lowercase, remove stop words
# (This will be done as part of the CountVectorizer)

# Feature extraction using Bag of Words
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
train_vectors = vectorizer.fit_transform(train_text)
val_vectors = vectorizer.transform(val_text)

# Tune the alpha parameter using cross-validation
alphas = [2**i for i in range(1)]  # Alpha values as powers of 2
best_alphas = {}
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    best_alpha = None
    best_score = 0
    scores_list = []
    for alpha in alphas:
        classifier = MultinomialNB(alpha=alpha)
        scores = cross_val_score(classifier, train_vectors, train_toxic[label], cv=3, scoring='accuracy')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
        scores_list.append(mean_score)
    best_alphas[label] = best_alpha
    print(f"{label}: Best alpha = {best_alpha:.4f}")

    # Print cross-validation scores for each alpha
    print(f"{label}: Cross-validation scores for each alpha:")
    for alpha, score in zip(alphas, scores_list):
        print(f"  Alpha = {alpha:.4f}, Score = {score:.4f}")

# Train Naive Bayes classifiers for each label using the best alpha
classifiers = {}
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    classifiers[label] = MultinomialNB(alpha=best_alphas[label])
    classifiers[label].fit(train_vectors, train_toxic[label])

# Evaluate the performance of the classifiers
print("Performance on the validation set:")
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    predictions = classifiers[label].predict(val_vectors)
    accuracy = accuracy_score(val_toxic[label], predictions)
    precision = precision_score(val_toxic[label], predictions)
    print(f"{label}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}")

    # Print the confusion matrix
    cm = confusion_matrix(val_toxic[label], predictions)
    print(f"{label}: Confusion Matrix = \n{cm}")

# Load the test data
test_df = pd.read_csv('data/test.csv')

# Define toxicity columns
toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load the test labels
test_labels_df = pd.read_csv('data/test_labels.csv')

# Get the IDs from the test set
test_ids = test_df['id']

# Filter test_labels_df to keep only rows with IDs present in test_df
test_labels_df = test_labels_df[test_labels_df['id'].isin(test_ids)]

# Drop duplicate IDs from test_labels_df, keeping the first occurrence.
# This is crucial to ensure that each test ID has only one corresponding label row.
test_labels_df = test_labels_df.drop_duplicates(subset=['id'], keep='first')

# Adjust test labels by replacing -1 with 0 (assuming -1 means not applicable or not toxic)
test_labels_df[toxicity_columns] = test_labels_df[toxicity_columns].replace(-1, 0)

# Merge test_df with the filtered, de-duplicated, and adjusted test_labels_df
# Using 'inner' merge ensures we only keep IDs present in both dataframes.
# Since test_labels_df is now de-duplicated and filtered by test_df's IDs,
# this merge should result in exactly the same number of rows as test_df.
test_df_merged = pd.merge(test_df, test_labels_df, on='id', how='inner')

# Verify the number of samples in the merged test set
print(f"Number of samples in the merged test set: {len(test_df_merged)}")

# Transform the test data using the same vectorizer
# Use the merged dataframe for comment_text
test_vectors = vectorizer.transform(test_df_merged['comment_text'])

# Predict the toxicity labels for the test data
test_predictions = {}
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    test_predictions[label] = classifiers[label].predict(test_vectors)

# Evaluate the performance of the classifiers on the test set
print("\nPerformance on the test set:")
for label in toxicity_columns:
    # Use adjusted true labels from test_df_merged
    true_labels = test_df_merged[label]
    predictions = test_predictions[label]

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    print(f"{label}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}")

    # Print the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"{label}: Confusion Matrix = \n{cm}")

# Extract and analyze word likelihoods
print("\nWord Likelihoods:")
feature_names = vectorizer.get_feature_names_out()

for label in toxicity_columns:
    print(f"\n--- {label} ---")
    # Get the log probability of each feature for the current classifier
    # feature_log_prob_ is shape (n_classes, n_features)
    # We need to find the index for class 1 (positive class)
    class_index = classifiers[label].classes_.tolist().index(1)
    log_prob_for_class = classifiers[label].feature_log_prob_[class_index]
    
    # Get indices sorted by log probability in descending order
    sorted_indices = np.argsort(log_prob_for_class)[::-1]
    
    # Get the top 10 words and their log probabilities
    top_words = [feature_names[i] for i in sorted_indices[:10]]
    top_log_probs = [log_prob_for_class[i] for i in sorted_indices[:10]]
    
    # Print the top words and their log probabilities
    for word, log_prob_val in zip(top_words, top_log_probs):
        print(f"  {word}: {log_prob_val:.4f}")


# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id']})
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    submission_df[label] = test_predictions[label]

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")
