import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

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
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_text)
val_vectors = vectorizer.transform(val_text)

# Tune the alpha parameter using cross-validation
alphas = [2**i for i in range(11)]  # Alpha values as powers of 2
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

# Transform the test data using the same vectorizer
test_vectors = vectorizer.transform(test_df['comment_text'])

# Predict the toxicity labels for the test data
test_predictions = {}
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    test_predictions[label] = classifiers[label].predict(test_vectors)

# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id']})
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    submission_df[label] = test_predictions[label]

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")
