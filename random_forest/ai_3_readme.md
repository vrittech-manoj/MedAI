# Disease Prediction System: Usage Guide

This guide explains how to use the disease prediction system based on your dataset structure.

## Dataset Requirements

Your dataset should have the following structure:
- Disease columns: Either `diseases_name` or `Associated Disease`
- Symptom columns: Named `symptoms_1`, `symptoms_2`, etc.
- Cause columns: Named `causes_1`, `causes_2`, etc.
- Additional health factors: `age_of_onset`, `genetic_factors`, `family_history`

## Setup Instructions

1. **Install Required Libraries**:
   ```
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib
   ```

2. **Prepare Your Dataset**:
   - Ensure your CSV file has the required columns
   - Clean any inconsistent data
   - Save as a CSV file

## Usage Instructions

### 1. Basic Command-Line Usage

```bash
# Train a new model
python disease_prediction.py --train --dataset your_dataset.csv --