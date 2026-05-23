import os
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, isnan, when, count

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)
app.secret_key = 'bigdata-secret-2024'

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
PLOTS_FOLDER  = 'static/plots'
ALLOWED_EXT   = {'csv'}

for d in [UPLOAD_FOLDER, MODELS_FOLDER, PLOTS_FOLDER]:
    os.makedirs(d, exist_ok=True)

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

spark          = None
current_df     = None
current_fname  = None


def get_spark():
    global spark
    if spark is None:
        os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
        spark = (SparkSession.builder
                 .appName("BigDataLab")
                 .master("local[*]")
                 .config("spark.driver.memory", "4g")
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
    return spark


def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def get_numeric_cols(df):
    return [f.name for f in df.schema.fields
            if str(f.dataType) in (
                'IntegerType()', 'LongType()',
                'DoubleType()', 'FloatType()', 'ShortType()')]


# ─────────────────────────────────────────────
#  ГЛАВНАЯ / ЗАГРУЗКА
# ─────────────────────────────────────────────
@app.route('/')
def index():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    return render_template('index.html', files=files, current_file=current_fname)


@app.route('/upload', methods=['POST'])
def upload():
    global current_df, current_fname
    if 'file' not in request.files:
        flash('Файл не выбран', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '' or not allowed(file.filename):
        flash('Выберите CSV-файл', 'error')
        return redirect(url_for('index'))

    fname    = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(filepath)

    try:
        sp         = get_spark()
        current_df = sp.read.csv(filepath, header=True, inferSchema=True)
        current_fname = fname
        rows = current_df.count()
        cols = len(current_df.columns)
        flash(f'Загружено «{fname}»: {rows:,} строк, {cols} столбцов', 'success')
    except Exception as e:
        flash(f'Ошибка чтения: {e}', 'error')

    return redirect(url_for('index'))


@app.route('/load/<filename>')
def load_file(filename):
    global current_df, current_fname
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        flash('Файл не найден', 'error')
        return redirect(url_for('index'))
    sp = get_spark()
    current_df    = sp.read.csv(filepath, header=True, inferSchema=True)
    current_fname = filename
    flash(f'Активен файл «{filename}»: {current_df.count():,} строк', 'success')
    return redirect(url_for('index'))


# ─────────────────────────────────────────────
#  АНАЛИЗ
# ─────────────────────────────────────────────
@app.route('/analyze')
def analyze():
    if current_df is None:
        flash('Сначала загрузите данные', 'error')
        return redirect(url_for('index'))

    schema = [{'name': f.name, 'type': str(f.dataType), 'nullable': f.nullable}
              for f in current_df.schema.fields]

    total  = current_df.count()
    miss   = {}
    for c in current_df.columns:
        dtype = str(current_df.schema[c].dataType)
        if 'Double' in dtype or 'Float' in dtype:
            n = current_df.filter(col(c).isNull() | isnan(col(c))).count()
        else:
            n = current_df.filter(col(c).isNull()).count()
        miss[c] = {'count': n, 'percent': round(n / total * 100, 2) if total else 0}

    desc_pd  = current_df.describe().toPandas()
    desc_rec = desc_pd.to_dict('records')
    desc_col = desc_pd.columns.tolist()

    preview  = current_df.limit(20).toPandas().to_dict('records')

    return render_template('analyze.html',
                           filename=current_fname,
                           schema=schema,
                           total_rows=total,
                           missing=miss,
                           describe=desc_rec,
                           desc_columns=desc_col,
                           rows=preview,
                           columns=current_df.columns)


# ─────────────────────────────────────────────
#  ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────
@app.route('/visualize')
def visualize():
    if current_df is None:
        flash('Сначала загрузите данные', 'error')
        return redirect(url_for('index'))
    num_cols = get_numeric_cols(current_df)
    plots    = os.listdir(PLOTS_FOLDER)
    return render_template('visualize.html',
                           filename=current_fname,
                           numeric_cols=num_cols,
                           plots=plots)


@app.route('/visualize/histogram', methods=['POST'])
def histogram():
    column = request.form.get('column')
    pdf    = current_df.select(column).dropna().limit(100000).toPandas()
    plt.figure(figsize=(10, 6))
    plt.hist(pdf[column], bins=50, edgecolor='black', alpha=0.7, color='#2196F3')
    plt.title(f'Гистограмма: {column}', fontsize=14)
    plt.xlabel(column); plt.ylabel('Частота'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, 'histogram.png'), dpi=100)
    plt.close()
    flash('Гистограмма построена', 'success')
    return redirect(url_for('visualize'))


@app.route('/visualize/scatter', methods=['POST'])
def scatter():
    cx = request.form.get('col_x')
    cy = request.form.get('col_y')
    pdf = current_df.select(cx, cy).dropna().limit(50000).toPandas()
    plt.figure(figsize=(10, 6))
    plt.scatter(pdf[cx], pdf[cy], alpha=0.3, s=10, color='#FF5722')
    plt.title(f'{cx} vs {cy}', fontsize=14)
    plt.xlabel(cx); plt.ylabel(cy); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, 'scatter.png'), dpi=100)
    plt.close()
    flash('Scatter plot построен', 'success')
    return redirect(url_for('visualize'))


@app.route('/visualize/correlation', methods=['POST'])
def correlation():
    num_cols = get_numeric_cols(current_df)
    if len(num_cols) < 2:
        flash('Нужно минимум 2 числовых столбца', 'error')
        return redirect(url_for('visualize'))

    df_clean   = current_df.select(num_cols).dropna()
    assembler  = VectorAssembler(inputCols=num_cols, outputCol='corr_vec')
    df_vec     = assembler.transform(df_clean).select('corr_vec')
    matrix     = Correlation.corr(df_vec, 'corr_vec').collect()[0][0].toArray()
    corr_pd    = pd.DataFrame(matrix, columns=num_cols, index=num_cols)

    size = max(10, len(num_cols))
    plt.figure(figsize=(size, size * 0.7))
    sns.heatmap(corr_pd, annot=True, cmap='RdYlGn', center=0, fmt='.2f', square=True)
    plt.title('Матрица корреляции (PySpark)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, 'correlation.png'), dpi=100)
    plt.close()
    flash('Матрица корреляции построена', 'success')
    return redirect(url_for('visualize'))


# ─────────────────────────────────────────────
#  ОБУЧЕНИЕ МОДЕЛИ
# ─────────────────────────────────────────────
@app.route('/train', methods=['GET', 'POST'])
def train():
    if current_df is None:
        flash('Сначала загрузите данные', 'error')
        return redirect(url_for('index'))

    num_cols = get_numeric_cols(current_df)
    results  = None

    if request.method == 'POST':
        target    = request.form.get('target')
        features  = request.form.getlist('features')
        mtype     = request.form.get('model_type', 'random_forest')
        test_size = float(request.form.get('test_size', 0.2))

        if not target or not features:
            flash('Выберите target и признаки', 'error')
            return redirect(url_for('train'))

        try:
            sel = current_df.select(features + [target]).dropna()
            total = sel.count()
            if total > 200000:
                sel = sel.sample(fraction=200000 / total, seed=42)

            pdf = sel.toPandas()
            X   = pdf[features]
            y   = pdf[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)

            models_map = {
                'random_forest':      (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 'Random Forest'),
                'logistic_regression':(LogisticRegression(max_iter=1000, random_state=42),                  'Logistic Regression'),
                'gradient_boosting':  (GradientBoostingClassifier(n_estimators=100, random_state=42),       'Gradient Boosting'),
            }
            model, mname = models_map.get(mtype, models_map['random_forest'])

            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 2)

            y_pred   = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix — {mname}')
            plt.ylabel('Истинное'); plt.xlabel('Предсказанное')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER, 'confusion_matrix.png'), dpi=100)
            plt.close()

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                fi = pd.DataFrame({'feature': features,
                                   'importance': model.feature_importances_}
                                  ).sort_values('importance', ascending=True)
                plt.figure(figsize=(10, max(4, len(features) * 0.5)))
                plt.barh(fi['feature'], fi['importance'], color='#4CAF50')
                plt.title(f'Важность признаков — {mname}')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_FOLDER, 'feature_importance.png'), dpi=100)
                plt.close()

            with open(os.path.join(MODELS_FOLDER, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)

            results = {'model_name': mname, 'accuracy': accuracy,
                       'train_time': elapsed,
                       'train_size': len(X_train), 'test_size': len(X_test)}
            flash(f'{mname} обучена! Accuracy: {accuracy}%', 'success')

        except Exception as e:
            flash(f'Ошибка: {e}', 'error')

    return render_template('train.html',
                           filename=current_fname,
                           columns=current_df.columns,
                           numeric_cols=num_cols,
                           results=results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
