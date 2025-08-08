# Salary Prediction with Hitters Dataset 🎯

This project predicts baseball player salaries using the **Hitters Dataset**.  
It demonstrates **data preprocessing, exploratory data analysis (EDA), feature engineering, and regression modeling** in Python.

---

## 📂 Project Structure

```
.
├── Salary Prediction with Hitters Dataset.ipynb  # Main Jupyter notebook
├── Hitters.csv                                   # Dataset
├── salary_prediction_notebook_structure.json     # Notebook structure metadata
```

---

## 📊 Dataset Overview

- **Source:** Hitters Dataset (Baseball player statistics and salaries)
- **Rows:** ~322
- **Columns:** Player statistics such as AtBat, Hits, HmRun, Runs, RBI, Walks, Years, Salary.

---

## 🛠️ Steps Performed

### 1. Data Loading & Cleaning
- Loaded dataset using `pandas`
- Handled missing values in `Salary` column
- Converted categorical features into numerical

**Example:**
```python
import pandas as pd

df = pd.read_csv("Hitters.csv")
df.dropna(subset=["Salary"], inplace=True)
df = pd.get_dummies(df, drop_first=True)
```

---

### 2. Exploratory Data Analysis (EDA)
- Visualized salary distribution
- Checked correlations between features and salary
- Compared salary trends across positions

**Visual Example:**  
Salary Distribution
|----------------------|
<img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/638be6d2-48c5-4ca8-a470-1a1293db2323" />




### 3. Feature Engineering
- Standardized numerical features
- Encoded categorical variables
- Removed multicollinear features

---

### 4. Model Training
- Used **Linear Regression** and **Ridge/Lasso Regression**
- Evaluated using RMSE, MAE, R²

**Example:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, predictions, squared=False))
print("R²:", r2_score(y_test, predictions))
```

---

### 5. Results
| Model              | RMSE  | R²    |
|--------------------|-------|-------|
| Linear Regression  | 100.5 | 0.89  |
| Ridge Regression   | 98.3  | 0.90  |
| Lasso Regression   | 102.1 | 0.88  |

---

## 📈 Visualizations
1. Salary Distribution
   
   <img width="565" height="455" alt="notebook_image_1" src="https://github.com/user-attachments/assets/c121bc55-a2ac-4295-bfa5-8649c133af23" />


3. Correlation Heatmap
   
   <img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/9f753e54-3665-4d5a-a961-9877da291754" />

5. Predicted vs Actual Salary  

  <img width="565" height="455" alt="notebook_image_4" src="https://github.com/user-attachments/assets/5600ddd7-e87a-416f-b6ef-c49e91f60c21" />


## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/salary-prediction-hitters.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook:
```bash
jupyter notebook "Salary Prediction with Hitters Dataset.ipynb"
```

---

## 📌 Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

---

## 📜 License
This project is licensed under the MIT License.
