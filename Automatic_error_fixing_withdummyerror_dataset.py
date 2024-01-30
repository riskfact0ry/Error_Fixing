import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the synthetic error dataset from CSV
csv_path = 'yourpath/synthetic_error_dataset.csv'
df = pd.read_csv(csv_path)

# Enhanced Data Preprocessing with Named Entity Recognition (NER)
def extract_named_entities(text):
    # Implement NER techniques to extract named entities (e.g., persons, organizations, locations)
    # Replace the following line with an actual NER implementation
    named_entities = ['<person>', '<organization>', '<location>']
    for entity in named_entities:
        text = text.replace(entity, '')
    return text

df['Processed Message'] = df['Error Message'].apply(extract_named_entities)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['Processed Message'], df['Error Type'], test_size=0.2, random_state=42
)

# Vectorization, feature selection, and RandomForestClassifier pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('feature_selector', SelectKBest(chi2)),
    ('classifier', RandomForestClassifier())
])

# Hyperparameter tuning using grid search
param_grid = {
    'vectorizer__max_features': [5000, 10000, 15000],
    'feature_selector__k': [100, 500, 1000],
    'classifier__n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Make predictions on test data
predictions = grid_search.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
