# Error_Fixing
Code for fixing the error while coding


Importing Libraries:

python

import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import spacy

    re: The re module provides regular expression matching operations.
    pandas: A powerful data manipulation library.
    sklearn: The scikit-learn library for machine learning tools.
    spacy: An NLP library used for Named Entity Recognition (NER).

Loading Synthetic Error Dataset:

python

csv_path = '/home/adminisrator/Desktop/DBS/charttracker/venv/synthetic_error_dataset.csv'
df = pd.read_csv(csv_path)

    Loads a synthetic error dataset from a CSV file into a pandas DataFrame (df).

Named Entity Recognition (NER) using SpaCy:

python

nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    return ' '.join(named_entities)

df['Processed Message'] = df['Error Message'].apply(extract_named_entities)

    Uses SpaCy to perform Named Entity Recognition on the 'Error Message' column and creates a new column 'Processed Message' with the extracted entities.

Data Splitting:

python

X_train, X_test, y_train, y_test = train_test_split(
    df['Processed Message'], df['Error Type'], test_size=0.2, random_state=42
)

    Splits the data into training and testing sets.

Pipeline Definition:

python

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=10000)),
    ('feature_selector', SelectKBest(chi2, k=500)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

    Defines a pipeline with three steps: TF-IDF Vectorization, Feature Selection, and Random Forest Classification.

Hyperparameter Tuning with Grid Search:

python

param_grid = {
    'vectorizer__max_features': [5000, 10000, 15000],
    'feature_selector__k': [100, 500, 1000],
    'classifier__n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

    Performs hyperparameter tuning using Grid Search with specified parameter grids.

Model Evaluation:

python

predictions = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

    Makes predictions on the test set and evaluates the classifier using accuracy and classification report.

Printing Results:

python

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

    Prints the best parameters found by Grid Search, accuracy, and classification report.
