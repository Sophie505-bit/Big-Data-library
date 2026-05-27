import pandas as pd
import numpy as np
import os

def generate_library_dataset(n=2_000_000, output_path='uploads/library_loans.csv'):
    np.random.seed(42)

    reader_ids = np.random.randint(1000, 50000, n)
    book_ids = np.random.randint(1, 15000, n)

    genres = np.random.choice(
        ['fiction', 'science', 'history', 'children', 'tech', 'philosophy', 'art', 'medicine'],
        n
    )

    year_published = np.random.randint(1950, 2025, n)
    loan_period_days = np.random.choice([14, 21, 30], n)
    reader_age = np.random.randint(7, 85, n)
    reader_total_loans = np.random.randint(0, 200, n).astype(float)
    reader_prev_overdue = np.random.randint(0, 20, n).astype(float)
    distance_km = np.random.exponential(3, n).round(1)
    copies_available = np.random.randint(1, 10, n)

    # вероятность просрочки зависит от 4 факторов
    overdue_score = (
        (reader_prev_overdue > 3).astype(float) * 0.4 +
        (distance_km > 5).astype(float) * 0.3 +
        (reader_age < 18).astype(float) * 0.15 +
        (loan_period_days == 14).astype(float) * 0.15
    )
    is_overdue = (overdue_score > np.random.uniform(0, 1, n)).astype(int)

    # 10% шума
    noise_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
    is_overdue[noise_idx] = 1 - is_overdue[noise_idx]

    days_overdue = np.where(is_overdue == 1, np.random.randint(1, 90, n), 0).astype(float)
    fine_amount = days_overdue * 10

    df = pd.DataFrame({
        'reader_id': reader_ids,
        'book_id': book_ids,
        'genre': genres,
        'year_published': year_published,
        'loan_period_days': loan_period_days,
        'reader_age': reader_age,
        'reader_total_loans': reader_total_loans,
        'reader_prev_overdue': reader_prev_overdue,
        'distance_km': distance_km,
        'copies_available': copies_available,
        'days_overdue': days_overdue,
        'fine_amount': fine_amount,
        'is_overdue': is_overdue,
    })

    # пропуски как в реальных данных
    nan_idx_1 = np.random.choice(n, size=int(n * 0.02), replace=False)
    nan_idx_2 = np.random.choice(n, size=int(n * 0.03), replace=False)
    df.loc[nan_idx_1, 'reader_total_loans'] = np.nan
    df.loc[nan_idx_2, 'distance_km'] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    generate_library_dataset()
