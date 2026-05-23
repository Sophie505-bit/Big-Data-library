import pandas as pd
import numpy as np
import os

def generate_dataset(n=2_000_000, output_path='uploads/test_data.csv'):
    np.random.seed(42)

    df = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n) * 2 + 1,
        'feature_3': np.random.randint(0, 100, n).astype(float),
        'feature_4': np.random.exponential(2, n),
        'feature_5': np.random.uniform(-10, 10, n),
        'category':  np.random.choice(['A', 'B', 'C'], n),
        'target':    np.random.randint(0, 2, n),
    })

    # target зависит от feature_1 и feature_3 — модель сможет это уловить
    mask = (df['feature_1'] > 0) & (df['feature_3'] > 50)
    df.loc[mask, 'target'] = 1
    df.loc[~mask, 'target'] = 0

    # добавляем шум 10% чтобы accuracy не была 100%
    noise_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
    df.loc[noise_idx, 'target'] = 1 - df.loc[noise_idx, 'target']

    # добавляем пропуски в два столбца
    nan_idx_1 = np.random.choice(n, size=int(n * 0.02), replace=False)
    nan_idx_2 = np.random.choice(n, size=int(n * 0.03), replace=False)
    df.loc[nan_idx_1, 'feature_2'] = np.nan
    df.loc[nan_idx_2, 'feature_4'] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    generate_dataset()
