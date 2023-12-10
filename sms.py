import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
encodings = ['utf-8', 'latin-1']
for encoding in encodings:
    try:
        data = pd.read_csv('spam.csv', encoding=encoding)
        # If successful, break out of the loop
        break
    except UnicodeDecodeError:
        print(f"Failed to decode using {encoding} encoding. Trying the next one.")
# Display the first few rows of the dataset
print(data.head())
# Convert labels to binary (0 for non-spam, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
# Example usage
new_sms = ["Free Viagra now!", "Hi, how are you?"]
new_sms_vectorized = vectorizer.transform(new_sms)
predictions_new = classifier.predict(new_sms_vectorized)
print(predictions_new)
