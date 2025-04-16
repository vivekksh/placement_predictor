# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset for NIT Silchar students with additional parameters
n_samples = 1000  # Increased sample size for more robust training
data = {
    'CGPA': np.random.uniform(6, 10, n_samples),  # Academic performance
    'Internships': np.random.binomial(1, 0.6, n_samples),  # Internship experience
    'Coding_Score': np.random.uniform(0, 100, n_samples),  # Coding test score
    'Alumni_Referral': np.random.binomial(1, 0.4, n_samples),  # Alumni referral status
    'Interview_Score': np.random.uniform(0, 100, n_samples),  # Mock interview score
    'Communication_Skills': np.random.randint(0, 6, n_samples),  # Soft skills rating
    'Project_Experience': np.random.randint(0, 4, n_samples),  # Number of projects
    'Department': np.random.choice(['CSE', 'ECE', 'ME', 'CE', 'EE'], n_samples),  # Departments
    'Placement_Season': np.random.choice(['7th Sem', '8th Sem', 'Pre-Final'], n_samples)  # Placement timing
}

df = pd.DataFrame(data)

# Define placement probability based on weighted features
df['Placement_Probability'] = (
    0.25 * (df['CGPA'] / 10) +  # 25% weight to CGPA
    0.15 * df['Internships'] +  # 15% weight to internships
    0.15 * (df['Coding_Score'] / 100) +  # 15% weight to coding skills
    0.10 * df['Alumni_Referral'] +  # 10% weight to referrals
    0.15 * (df['Interview_Score'] / 100) +  # 15% weight to interview score
    0.10 * (df['Communication_Skills'] / 5) +  # 10% weight to communication
    0.10 * df['Project_Experience']  # 10% weight to project experience
)
df['Placement'] = (df['Placement_Probability'] > 0.7).astype(int)  # Threshold for placement

# Display the first few rows of the dataset
print("Sample Dataset:")
print(df.head())

# Preprocess the data
# Define categorical and numerical columns
categorical_features = ['Department', 'Placement_Season']
numerical_features = ['CGPA', 'Internships', 'Coding_Score', 'Alumni_Referral',
                      'Interview_Score', 'Communication_Skills', 'Project_Experience']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Prepare features and target
X = df.drop(columns=['Placement', 'Placement_Probability'])
y = df['Placement']

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("\nModel Evaluation (Test Set):")
print(f'Accuracy: {accuracy:.2f}')
print(f'ROC-AUC Score: {roc_auc:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:")
print(f'Average Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')

# Feature importance (from Random Forest)
feature_names = (numerical_features +
                 pipeline.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(categorical_features).tolist())
importances = pipeline.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Save the pipeline (includes preprocessor and model)
dump(pipeline, 'placement_model_pipeline.joblib')
print("Model pipeline saved successfully.")