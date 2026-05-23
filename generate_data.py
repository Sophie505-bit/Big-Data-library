import pandas as pd
import numpy as np
import os

os.makedirs('uploads', exist_ok=True)

n = 2_000_000
np.random.seed(42)

print(f"Генерируем {n:,} строк...")

df = pd.DataFrame({
    'feature_1': np.random.randn(n),
    'feature_2': np.random.randn(n) * 2 + 1,
    'feature_3': np.random.randint(0, 100, n),
    'feature_4': np.random.exponential(2, n),
    'feature_5': np.random.uniform(-10, 10, n),
    'target':    np.zeros(n, dtype=int)
})

mask = (df['feature_1'] > 0) & (df['feature_3'] > 50)
df.loc[mask,  'target'] = 1
df.loc[~mask, 'target'] = 0

noise = np.random.choice(n, size=int(n * 0.1), replace=False)
df.loc[noise, 'target'] = 1 - df.loc[noise, 'target']

out = 'uploads/test_data.csv'
df.to_csv(out, index=False)
mb = os.path.getsize(out) / 1_048_576
print(f"Готово! Файл: {out}")
print(f"Размер: {mb:.1f} МБ | Строк: {len(df):,} | Столбцов: {len(df.columns)}")
