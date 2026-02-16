# 🚴 Explainability of Machine Learning Models
## Comparing SHAP and Permutation Feature Importance on Bike Sharing Demand Prediction


## 📋 Project Overview

This project explores **explainability techniques** for machine learning models predicting hourly bike-sharing demand. The main goal is to compare two popular explainability methods:

-  **Permutation Feature Importance (PFI)** - Global feature importance via performance degradation
-  **SHAP (SHapley Additive exPlanations)** - Game theory-based local and global explanations

Two ensemble regression models (**Random Forest** and **XGBoost**) are trained and evaluated, with comprehensive explainability analysis to understand which features drive predictions and how different methods compare.

---

##  Key Results

### **Model Performance**

| Model | R² | MAE (bikes) | RMSE (bikes) |
|-------|-----|-------------|--------------|
| **Random Forest** | 0.890 | 48.03 | 73.15 |
| **XGBoost** ⭐ | **0.912** | **42.53** | **65.22** |

**XGBoost selected as production model** - achieves 91.2% explained variance with 19% lower MAE than baseline.

---

### **Feature Importance - Top 5**

Both PFI and SHAP consistently identify:

1. **`hr`** (hour of day) - Dominant predictor (~35-45% importance)
2. **`hour_sin`** (cyclical hour encoding) - Critical temporal pattern
3. **`hour_cos`** (cyclical hour encoding) - Complementary temporal signal
4. **`temp_workingday`** (interaction feature) - Weather × workday pattern
5. **`temp`** (temperature) - Environmental factor

**SHAP vs PFI Correlation:** 0.770 (good agreement with complementary perspectives)

---

### **Error Analysis Highlights**

- **70.6%** of predictions within ±50 bikes (good quality)
- **45.5%** of predictions within ±20 bikes (excellent quality)
- Model shows **slight underestimation bias** (-13.9 bikes on average)
- **Peak hours** (8am, 5-7pm) have highest errors due to high variability
- Performance improves with demand level (better % accuracy for high demand)

---

### **Fairness Analysis**

Model performance varies slightly across groups but remains production-ready:

| Group Type | Variance | Assessment |
|------------|----------|------------|
| **Seasons** | 4-5 bikes MAE range | ⚠️ Minor variation (summer best) |
| **Time of day** | Consistent | ✅ Reliable across periods |
| **Weather** | Stable for common conditions | ⚠️ Limited data for extremes |
| **Day type** | Balanced | ✅ Weekday/weekend similar |

**Key findings:**
- Coefficient of variation: <15% (acceptable threshold)
- No systematic bias against specific groups
- Variations correlate with data distribution (summer has more samples)

**Conclusion:** Model is **fair enough for production** - minor variations exist but do not indicate problematic bias.
---

## 🛠️ Technologies Used

**Core ML Stack:**
- Python 3.10+
- scikit-learn - Random Forest, PFI, metrics
- XGBoost - Gradient boosting
- SHAP - Explainability analysis

**Data & Visualization:**
- Pandas & NumPy - Data manipulation
- Matplotlib & Seaborn - Visualization

**Environment:**
- Google Colab - Development & execution

---

## 📊 Dataset

**Source:** [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) (Kaggle)

**Description:**
- **17,379 hourly observations** (2011-2012)
- **Target:** `cnt` (total bike rentals per hour)
- **Features:** Weather conditions, temporal attributes, derived features

**Key Features:**
- Temporal: `hr`, `weekday`, `mnth`, `season`, `yr`, `workingday`, `holiday`
- Weather: `temp`, `atemp`, `hum`, `windspeed`, `weathersit`
- Engineered: Cyclical encodings (`hour_sin/cos`, `month_sin/cos`, `weekday_sin/cos`)
- Interactions: `temp_hour`, `temp_workingday`, `humidity_temp`

---

## 🔬 Methodology

### **1. Feature Engineering**

**Cyclical Encoding** (for temporal features):
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
**Rationale:** Hour 23 and hour 0 are numerically far (23 units) but temporally close (1 hour). Cyclical encoding captures this circular nature.

**Interaction Features:**
- `temp_hour` - Captures "warm afternoon" vs "cold night" patterns
- `temp_workingday` - Weekend leisure vs weekday commute dynamics
- `humidity_temp` - "Muggy" discomfort effect

**Impact:** +5% R² improvement from feature engineering

---

### **2. Train/Test Split**

**Time-based split (80/20)** - Preserves temporal order to simulate real forecasting:
- Training: First 80% chronologically (13,903 samples)
- Test: Last 20% chronologically (3,476 samples)

**Rationale:** Random split would leak future information; time-based split tests true forecasting ability.

---

### **3. Model Training**

**Random Forest:**
```python
n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2
```

**XGBoost:**
```python
n_estimators=300, learning_rate=0.05, max_depth=6, 
subsample=0.8, colsample_bytree=0.8
```

**Key difference:** RF uses deep independent trees (bagging); XGBoost uses shallow sequential trees (boosting).

---

### **4. Explainability Analysis**

**Permutation Feature Importance (PFI):**
- Shuffle each feature and measure performance drop
- Fast computation (~30 seconds for both models)
- Global importance ranking

**SHAP:**
- Game theory-based feature contribution
- Local (per-prediction) + Global (aggregated) explanations
- Computationally intensive (~2-3 minutes)
- Provides magnitude and direction of impact

**Comparison Strategy:**
- PFI for both models (validation of consistency)
- SHAP for best model only (detailed analysis)

---

## 💡 Key Insights

### **1. Temporal Features Dominate**

**Hour of day** (`hr`, `hour_sin`, `hour_cos`) accounts for **~70% of feature importance**:
- Morning peak (7-9am) - Commuters
- Evening peak (5-7pm) - Return commute + leisure
- Night valley (0-5am) - Minimal demand

**Cyclical encoding is critical** - `hour_sin` and `hour_cos` rank #2 and #3, validating the feature engineering approach.

---

### **2. SHAP vs PFI: Complementary Perspectives**

**Correlation: 0.770** (good but not perfect agreement)

**Where they agree:**
- Top 3 features identical (`hr`, `hour_sin`, `hour_cos`)
- Temporal features dominate
- Weather features secondary

**Where they differ:**
- **`yr` (year):** SHAP sees as important (0.81), PFI sees as negligible (0.02)
  - **Reason:** Other features (season, month) compensate when `yr` is shuffled (PFI), but SHAP measures direct contribution
- **Interaction features:** SHAP values higher (captures non-linear effects)

**Conclusion:** Methods offer **different but valid perspectives** - PFI measures "irreplaceability," SHAP measures "contribution."

---

### **3. Model Performance Trade-offs**

**XGBoost advantages:**
- 2.2% higher R² (0.912 vs 0.890)
- 11% lower MAE (42.5 vs 48.0 bikes)
- Better calibration (less bias)

**Why XGBoost wins:**
- Boosting corrects errors iteratively
- Regularization (subsample, colsample) prevents overfitting
- Gradient-based optimization more precise than averaging

---

### **4. Error Analysis Findings**

**Problem areas:**
- Peak hours (8am, 5-7pm) - High variance in demand
- Very low demand (<50 bikes) - High % error but low absolute error
- Extreme events not captured by features (holidays, strikes, etc.)

**Strengths:**
- Medium-high demand (100-400 bikes) - Strong performance
- 87.9% predictions within ±100 bikes
- Production-ready for typical scenarios

---

## 🚀 Reproducibility

### **Requirements**

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

### **Data Setup**

1. Download `hour.csv` from [Kaggle Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
2. Upload to Google Colab or set local path
3. Update data path in notebook if needed

### **Execution**

```python
# In Google Colab:
1. Mount Google Drive (if data stored there)
2. Run all cells sequentially
3. Total runtime: ~6-8 minutes
```

---

## 🎓 Educational Context

This project was developed for coursework focused on **Explainable AI (XAI)**. 

**Learning objectives achieved:**
- ✅ Understand difference between model-agnostic (PFI) and model-specific (SHAP) explainability
- ✅ Apply feature engineering for temporal data
- ✅ Compare ensemble methods (bagging vs boosting)
- ✅ Conduct comprehensive error and fairness analysis
- ✅ Interpret and communicate ML model behavior

**Skills demonstrated:**
- Feature engineering (domain knowledge application)
- Hyperparameter selection (justified choices)
- Model evaluation (multiple metrics, error analysis)
- Explainability techniques (PFI + SHAP)
- Production mindset (fairness, bias, robustness)

---

## 🔮 Future Enhancements

**Potential improvements:**

1. **Feature Expansion:**
   - Weather forecasts (next-day predictions)
   - Special events calendar (holidays, concerts, strikes)
   - Bike station locations (spatial features)

2. **Advanced Models:**
   - Temporal deep learning (LSTM, Transformers)
   - Ensemble stacking (RF + XGB + others)
   - Causal inference models

3. **Explainability Extensions:**
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Anchor explanations
   - Counterfactual explanations ("What if temp was 5°C higher?")

4. **Deployment:**
   - REST API for real-time predictions
   - Interactive Streamlit dashboard
   - Automated retraining pipeline

---

## 👤 Author

**Dora Šakić**  

---

## 📄 License

This project is for **educational purposes**. Dataset license follows Kaggle terms. Code is available for learning and non-commercial use.

---

**Completion date:** February 2026

**Final deliverables:**
- ✅ Complete Jupyter/Colab notebook with all analyses
- ✅ Comprehensive README documentation
- ✅ Error and fairness analysis
- ✅ Model comparison and explainability study

---

**⭐ If you found this project helpful, please consider giving it a star!**
