from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

def train_and_evaluate_classifier():
    # Generate a synthetic 10-class dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=10, random_state=42)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define a preprocessing pipeline
    numeric_features = np.arange(X.shape[1])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Create a pipeline that first preprocesses the data, then fits a classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier using F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

    # Print a detailed classification report
    print(classification_report(y_test, y_pred))

train_and_evaluate_classifier()