
# 🧠 AI-Powered Clinical Prediction: Stroke Risk Evaluation

## 📌 Abstract & Research Goal
Stroke is a leading cause of severe disability and mortality globally. Early diagnostic stratification using predictive Artificial Intelligence allows clinicians to identify vulnerable cohorts prior to acute events. This project constructs a **Machine Learning Classification Pipeline** to evaluate clinical features (hypertension, glucose profile, BMI, and age) for automated stroke risk scoring.

---

## 🔬 Clinical Feature Engineering
The predictive algorithm operates on key biostatistical indicators:
- **Age**: Patient demographic baseline.
- **Hypertension & Heart Disease**: Binary cardiovascular comorbidities (0 = Absent, 1 = Present).
- **Average Glucose Level**: Glycemic metric (mg/dL).
- **Body Mass Index (BMI)**: Obesity stratification index ($kg/m^2$).
- **Stroke Target**: Binary diagnostic output.

---

## ⚙️ AI Architecture & Machine Learning Stack
- **Algorithm**: Random Forest Ensemble Classifier (Scikit-Learn).
- **Data Pipeline**: Stratified Train-Test Split (80/20 ratio).
- **Metrics**: Accuracy, Precision, Recall, and Classification Metrics.

---

## 🛠️ Reproduction Instructions
To execute the AI pipeline locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/sama514/AI-Stroke-Risk-Prediction.git](https://github.com/sama514/AI-Stroke-Risk-Prediction.git)
