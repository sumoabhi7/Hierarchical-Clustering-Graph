import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
num_customers = 1000

data = {
    'CUST_ID': ['C' + str(10000 + i) for i in range(num_customers)],
    'BALANCE': np.round(np.random.uniform(0, 10000, num_customers), 2),
    'BALANCE_FREQUENCY': np.round(np.random.uniform(0.5, 1.0, num_customers), 2),
    'PURCHASES': np.round(np.random.uniform(10, 5000, num_customers), 2),
    'ONEOFF_PURCHASES': np.round(np.random.uniform(10, 3000, num_customers), 2),
    'INSTALLMENTS_PURCHASES': np.round(np.random.uniform(10, 2000, num_customers), 2),
    'CASH_ADVANCE': np.round(np.random.uniform(0, 5000, num_customers), 2),
    'PURCHASES_FREQUENCY': np.round(np.random.uniform(0, 0.8, num_customers), 2),
    'ONEOFF_PURCHASES_FREQUENCY': np.round(np.random.uniform(0, 0.5, num_customers), 2),
    'PURCHASES_INSTALLMENTS_FREQUENCY': np.round(np.random.uniform(0, 0.5, num_customers), 2),
    'CASH_ADVANCE_FREQUENCY': np.round(np.random.uniform(0, 0.3, num_customers), 2),
    'CASH_ADVANCE_TRX': np.random.poisson(1, num_customers),
    'PURCHASES_TRX': np.random.poisson(15, num_customers),
    'CREDIT_LIMIT': np.round(np.random.uniform(1000, 15000, num_customers), 2),
    'PAYMENTS': np.round(np.random.uniform(100, 6000, num_customers), 2),
    'MINIMUM_PAYMENTS': np.round(np.random.uniform(10, 2000, num_customers), 2),
    'PRC_FULL_PAYMENT': np.round(np.random.uniform(0, 0.5, num_customers), 2),
    'TENURE': np.random.randint(6, 36, num_customers)
}

# Create DataFrame and add missing values
df = pd.DataFrame(data)
for col in ['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Save to CSV
df.to_csv('synthetic_cc_data.csv', index=False)
print("Successfully generated synthetic_cc_data.csv")