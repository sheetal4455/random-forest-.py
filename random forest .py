import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ---------------------------
# Step 1: Create Dataset
# ---------------------------
data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Very happy with service",
        "Worst experience ever",
        "I hate this",
        "Very bad quality"
    ],
    "sentiment": ["Positive", "Positive", "Positive", "Negative", "Negative", "Negative"]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# ---------------------------
# Step 2: Convert Text to Numbers
# ---------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# ---------------------------
# Step 3: Random Forest Optimization
# ---------------------------
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(rf, param_grid, cv=2)
grid.fit(X, y)

print("\nBest Parameters:")
print(grid.best_params_)

# ---------------------------
# Step 4: Prediction
# ---------------------------
new_text = ["I am very happy"]
new_text_vector = vectorizer.transform(new_text)

prediction = grid.predict(new_text_vector)

print("\nPrediction for new text:")
print(prediction)
