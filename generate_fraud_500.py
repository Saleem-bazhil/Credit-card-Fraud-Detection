import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
np.random.seed(42)
ROWS = 500
FEATURES = 30
FRAUD_COUNT = 25   # 5% fraud (safe for SMOTE)

# =========================
# GENERATE FEATURES
# =========================
X = np.random.randn(ROWS, FEATURES)

# Create fraud labels
labels = np.zeros(ROWS, dtype=int)
fraud_idx = np.random.choice(ROWS, FRAUD_COUNT, replace=False)
labels[fraud_idx] = 1

# Inject fraud patterns
X[labels == 1, 0] += 3.0
X[labels == 1, 1] -= 2.0
X[labels == 1, 4] += 3.5
X[labels == 1, 7] -= 2.5

# =========================
# AMOUNT FEATURE
# =========================
amount = np.random.exponential(scale=100, size=ROWS)
amount[labels == 1] *= np.random.uniform(2, 5, FRAUD_COUNT)

# =========================
# CREATE DATAFRAME
# =========================
columns = [f"V{i}" for i in range(1, FEATURES + 1)]
df = pd.DataFrame(X, columns=columns)

df["Amount"] = amount
df["Amount_Normalized"] = np.log1p(amount)
df["Class"] = labels

# =========================
# ADD SMALL MISSING VALUES
# =========================
for col in np.random.choice(columns, size=5, replace=False):
    mask = np.random.rand(ROWS) < 0.01
    df.loc[mask, col] = np.nan

# =========================
# SAVE CSV
# =========================
df.to_csv("credit_card_fraud_500.csv", index=False)

print("✅ Dataset generated successfully!")
print(df["Class"].value_counts())
