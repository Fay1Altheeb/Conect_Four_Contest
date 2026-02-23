# Connect Four Move Prediction Using Machine Learning

A supervised machine learning approach to predicting the optimal next move in Connect Four using feature engineering and ensemble models.

---

## Overview

This project formulates the Connect Four game as a multi-class classification problem.

The objective is to predict the optimal column (0–6) given a board state.

The 6×7 board is flattened into 42 features encoded as:

- `+1` → Current player
- `-1` → Opponent
- `0`  → Empty cell

An additional feature represents the player's turn.

---

## Problem Formulation

**Input:**
- 42 board features (p1–p42)
- Turn indicator

**Output:**
- Optimal column index (0–6)

---

## Dataset

Pre-split into:

- `train.csv`
- `val.csv`
- `test.csv`

Test labels are withheld for blind evaluation.

---

## Preprocessing

- Standardization using `StandardScaler`
- Train/Validation/Test separation
- No data leakage during scaling

---

## Feature Engineering

Additional features were introduced to capture strategic patterns:

- Game state context  
  - Piece count  
  - Material advantage  
  - Board fill ratio  

- Positional control  
  - Center column control  
  - Edge control  
  - Bottom row control  

- Pattern detection  
  - Two-in-a-row  
  - Three-in-a-row threats  

- Critical move detection  
  - Immediate win detection  
  - Opponent win blocking  

- Advanced tactical features  
  - Fork detection  
  - Window-based board scoring  

---

## Models Implemented

### Baseline Models

- Logistic Regression  
- Decision Tree  
- Random Forest  

### Advanced Models

- Gradient Boosting  
- XGBoost  
- Multi-Layer Perceptron (MLP)  
- LightGBM  
- CatBoost  

---

## Results

| Model | Validation Accuracy |
|--------|--------------------|
| Logistic Regression | 40.2% |
| Decision Tree | 47.1% |
| Random Forest | 58% |
| Gradient Boosting | 68.25% |
| MLP | 63.97% |
| XGBoost | 70.89% |
| LightGBM | 70.69% |
| CatBoost | 71.15% |

**Final Test Accuracy (CatBoost): 69.8%**

CatBoost demonstrated the best generalization performance.

---

## Hyperparameter Tuning (CatBoost)

- Iterations: 500–800  
- Learning Rate: 0.03–0.05  
- Depth: 6–8  
- L2 Regularization  
- RandomizedSearchCV (3-fold cross-validation)  
- Early stopping (100 rounds)  

---

## Key Findings

- Ensemble models outperform simple linear models.
- Feature engineering significantly improves performance.
- XGBoost showed overfitting on the test set.
- CatBoost achieved the most stable and generalizable results.

---

## Project Structure

```
data/          # Dataset files
src/           # Model training scripts
notebooks/     # Experiments and analysis
docs/          # Full project report
```

---
