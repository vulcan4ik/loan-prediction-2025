# 🏦 Loan Payback Prediction

> Прогнозирование выплаты кредита с помощью CatBoost  
> **Kaggle Playground Series S5E11 | Score: 0.92317 (ROC-AUC)**  
> **ROC-AUC: 0.923 | F1-score: 0.944 | Recall: 0.89 | 25 признаков + 5-Fold CV**

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S5E11-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s5e11)
[![Score](https://img.shields.io/badge/Kaggle%20Score-0.92317-3DDC84?style=flat)]()

---

## 📌 Обзор

Проект посвящён **оценке вероятности возврата кредита** заёмщиком по финансовым и социальным признакам.  
Используется **CatBoost**, устойчивый к категориальным данным и дисбалансу классов.  

Модель демонстрирует **ROC-AUC = 0.923**, стабильно различая надёжных и рискованных клиентов.  
Создана в рамках соревнования **Kaggle Playground Series S5E11**.

---

## 🎯 Задача

**Бинарная классификация:**
- `1` — кредит выплачен  
- `0` — дефолт  

📊 **Дисбаланс классов:** 80% выплативших vs 20% невыплативших  
📁 **Размер данных:** 594k train / 148k test  

---

## 🔍 Методология

### 1. **EDA**
- Анализ распределений дохода, кредитного рейтинга и процентных ставок.  
- Проверка категориальных признаков: `employment_status`, `education_level`, `loan_purpose`.  
- Выявление выбросов методом IQR и PHIK-корреляция для категорий.

### 2. **Feature Engineering (12 новых признаков)**

```python
loan_to_income = loan_amount / annual_income
debt_interest_ratio = (loan_amount * (1 + interest_rate/100)) / annual_income
credit_risk_index = credit_score * (1 - debt_to_income_ratio)
loan_income_stress = loan_amount / (annual_income * debt_to_income_ratio)
log_annual_income = log1p(annual_income)
log_loan_amount = log1p(loan_amount)
```

Добавлены рейтинговые признаки (`grade_num`, `combined_rating`, `default_rate`),  
что позволило улучшить ROC-AUC на **+0.004**.

### 3. **Модель**
- CatBoost с балансировкой классов (`auto_class_weights='Balanced'`)  
- 5-Fold Stratified Cross-Validation  
- Оптимизация порога по F1-score (лучший порог = 0.20 → F1 = 0.9446)

---

## 📈 Финальные метрики

| Метрика | Значение |
|----------|-----------|
| Accuracy | 0.872 |
| F1-score | 0.944 |
| Recall | 0.892 |
| ROC-AUC | **0.923** |

---

### 🔝 Топ-10 признаков

| Признак | Важность |
|----------|-----------|
| credit_risk_index | 14.8% |
| debt_score_ratio | 13.6% |
| credit_score | 11.9% |
| debt_to_income_ratio | 11.5% |
| combined_rating | 9.8% |
| interest_rate | 7.2% |
| loan_income_stress | 6.5% |
| log_loan_amount | 5.9% |
| loan_to_income | 5.5% |
| annual_income | 4.6% |

---

## ⚙️ Эволюция модели

```
v1: Baseline CatBoost (без FE)         → ROC-AUC = 0.919
v2: + Feature Engineering               → ROC-AUC = 0.9228
v3: + Cross Validation (5-Fold)         → ROC-AUC = 0.9230
v4: + Порог F1 + proba submit           → ROC-AUC = 0.92317 🏆
```

---

## 📁 Структура проекта

```
loan-payback/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── ipynb/
│   └── loan-payback-catboost.ipynb
├── models/
│   └── catboost_loanpayback_final.cbm
├── output/
│   ├── feature_importance.png
│   └── submission_v3.csv
└── README.md
```

---

## 🚀 Быстрый старт

```bash
git clone https://github.com/vulcan4ik/loan-payback-prediction.git
cd loan-payback-prediction
pip install -r requirements.txt
jupyter lab ipynb/loan-payback-catboost.ipynb
```

---

## ✅ Ключевые техники

1. **Feature Engineering (12 новых признаков)**  
2. **PHIK корреляция для категорий**  
3. **Balanced CatBoost** для дисбаланса классов  
4. **Оптимизация порога по F1-score**  
5. **Кросс-валидация (5-Fold)** для стабильности

---

## 🔮 Дальнейшие улучшения

- [ ] Optuna для автоматического подбора параметров  
- [ ] SHAP анализ для интерпретации  
- [ ] Ансамбли (LightGBM + CatBoost)  
- [ ] Калибровка вероятностей  
- [ ] MLflow pipeline  

---

## 🧰 Технологии

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF?style=flat)

---

<div align="center">

**Kaggle Playground Series S5E11 | ROC-AUC = 0.92317 🏆**  
Модель прогнозирования возврата кредита на CatBoost

[🔗 Kaggle Leaderboard](https://www.kaggle.com/competitions/playground-series-s5e11)

</div>

