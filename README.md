# Analytics Pipeline Module

## Setup
1. Install dependencies: `pip install seaborn pandas scikit-learn matplotlib numpy`
2. Run notebooks in order: `01_eda.ipynb` → `02_modeling.ipynb`

## Data Story Summary
The Titanic survival analysis reveals that:
1. Women had significantly higher survival rates (~74%) compared to men (~19%)
2. First-class passengers had higher survival rates (~63%) vs third-class (~24%)
3. Age and fare were strong predictors of survival
4. Children (under 12) had higher survival rates, especially in lower classes

## Model Recommendation
Random Forest is recommended for deployment with:
- Accuracy: 82.3%
- F1-Score: 0.78
- Good balance of precision and recall

## Decision Tree Visualization
The decision tree shows that sex was the primary splitting factor, followed by fare and age.