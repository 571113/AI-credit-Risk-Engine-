# AI-credit-Risk-Engine-
Machine learning credit risk assessment, probability of default calibration, model explainability and risk analytics.
# AI Credit Risk Engine

A Python-based credit risk monitoring engine combining default probability estimation and anomaly detection to prioritize credit files for risk review.

## Project objectives

- Estimate the probability of credit default using a supervised machine learning model.
- Identify unusual borrower profiles with Isolation Forest.
- Combine both signals into four monitoring priorities.
- Evaluate the risk concentration and default capture rate across priority groups.

## Engine architecture

1. **Default risk scoring:** the fitted classification model estimates a probability of default for each file.
2. **Anomaly detection:** Isolation Forest identifies unusual profiles.
3. **Risk prioritization:** files are assigned to one of four categories:
   - Priority 1: elevated default risk and anomalous profile.
   - Priority 2: anomalous profile without elevated modelled default risk.
   - Priority 3: elevated modelled default risk.
   - Standard monitoring: neither condition is met.
4. **Portfolio reporting:** the engine summarizes file counts, average estimated default probability and anomalies by priority.

## Test-set demonstration

The current test-set evaluation contains 6,000 observations and 1,327 observed defaults, corresponding to a 22.12% default rate.

| Monitoring category | Files | Observed default rate | Share of all defaults |
|---|---:|---:|---:|
| Priority 1 | 81 | 79.01% | 4.82% |
| Priority 3 | 1,017 | 53.59% | 41.07% |
| Priority 2 | 219 | 15.98% | 2.64% |
| Standard monitoring | 4,683 | 14.58% | 51.47% |

Priorities 1 and 3 together cover 1,098 files (18.3% of the test set) and contain 609 observed defaults, or approximately 45.89% of all defaults in the test set.

These figures are a demonstration on one test sample, not a guarantee of future performance.

## Repository structure

```text
AI-credit-Risk-Engine/
├── README.md
├── requirements.txt
└── src/
    ├── credit_risk_engine.py
    ├── data_processing.py
    ├── model.py
    └── tests/
        ├── test_credit_risk_engine.py
        └── test_data_processing.py
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run the tests

From the repository root:

```bash
python -m pytest src/tests -v
```

## Important limitations

- Priority thresholds are configurable and should be validated against the institution's risk appetite.
- An anomaly is not necessarily a bad borrower or a default.
- Probability estimates require independent calibration and validation before production use.
- Monitoring should include discrimination, calibration, stability, fairness, and data-quality checks.
- This project is a technical demonstration and is not intended to make automated lending decisions.

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, pytest.
