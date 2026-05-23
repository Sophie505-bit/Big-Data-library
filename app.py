import os
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)
app.secret_key = 'bigdata-secret-key-2024'

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
PLOTS_FOLDER = 'static/plots'
ALLOWED_EXTENSIONS = {'csv'}

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, PLOTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# максимальный размер загружаемого файла — 500 мб
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# глобальные переменные для текущей сессии
spark = None
current_df = None
current_filename = None

def get_spark():
    global spark
    # spark создаётся один раз и переиспользуется
    if spark is None:
        os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
        spark = SparkSession.builder \
            .appName('BigDataLab') \
            .master('local[*]') \
            .config('spark.driver.memory', '4g') \
            .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \
            .getOrCreate()
        spark.sparkContext.setLogLevel('ERROR')
    return spark

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_numeric_cols(df):
    # возвращает только числовые столбцы датафрейма
    numeric_types = (
        'IntegerType()', 'LongType()',
        'DoubleType()', 'FloatType()', 'ShortType()'
    )
    return [f.name for f in df.schema.fields if str(f.dataType) in numeric_types]

@app.route('/')
def index():
    files = os.listdir(UPLOAD_FOLDER)
    csv_files = [f for f in files if f.endswith('.csv')]
    return render_template('index.html',
                           files=csv_files,
                           current_file=current_filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df, current_filename
    if 'file' not in request.files:
        flash('файл не выбран', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('файл не выбран', 'error')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            sp = get_spark()
            # pyspark читает csv с автоматическим определением типов
            current_df = sp.read.csv(filepath, header=True, inferSchema=True)
            current_filename = filename
            row_count = current_df.count()
            col_count = len(current_df.columns)
            flash(f'файл загружен: {row_count:,} строк, {col_count} столбцов', 'success')
        except Exception as e:
            flash(f'ошибка при чтении файла: {str(e)}', 'error')
        return redirect(url_for('index'))
    flash('допустимы только csv-файлы', 'error')
    return redirect(url_for('index'))

@app.route('/load/<filename>')
def load_existing(filename):
    global current_df, current_filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        flash('файл не найден', 'error')
        return redirect(url_for('index'))
    try:
        sp = get_spark()
        current_df = sp.read.csv(filepath, header=True, inferSchema=True)
        current_filename = filename
        row_count = current_df.count()
        flash(f'загружен: {row_count:,} строк', 'success')
    except Exception as e:
        flash(f'ошибка: {str(e)}', 'error')
    return redirect(url_for('index'))

@app.route('/analyze')
def analyze():
    if current_df is None:
        flash('сначала загрузите данные', 'error')
        return redirect(url_for('index'))

    # собираем схему: имя столбца, тип, nullable
    schema_info = []
    for field in current_df.schema.fields:
        schema_info.append({
            'name': field.name,
            'type': str(field.dataType),
            'nullable': field.nullable
        })

    rows = current_df.limit(20).toPandas().to_dict('records')
    columns = current_df.columns
    total_rows = current_df.count()

    from pyspark.sql.functions import col, isnan

    # считаем пропуски отдельно для float/double (isnan) и остальных (isNull)
    missing_data = {}
    for c in current_df.columns:
        dtype = str(current_df.schema[c].dataType)
        try:
            if 'Double' in dtype or 'Float' in dtype:
                miss = current_df.filter(col(c).isNull() | isnan(col(c))).count()
            else:
                miss = current_df.filter(col(c).isNull()).count()
        except Exception:
            miss = 0
        missing_data[c] = {
            'count': miss,
            'percent': round(miss / total_rows * 100, 2) if total_rows > 0 else 0
        }

    desc_pdf = current_df.describe().toPandas()
    desc_dict = desc_pdf.to_dict('records')

    return render_template('analyze.html',
                           filename=current_filename,
                           schema=schema_info,
                           rows=rows,
                           columns=columns,
                           total_rows=total_rows,
                           missing=missing_data,
                           describe=desc_dict,
                           desc_columns=desc_pdf.columns.tolist())

@app.route('/visualize')
def visualize():
    if current_df is None:
        flash('сначала загрузите данные', 'error')
        return redirect(url_for('index'))
    columns = current_df.columns
    numeric_cols = get_numeric_cols(current_df)
    plots_exist = {
        'histogram': os.path.exists(os.path.join(PLOTS_FOLDER, 'histogram.png')),
        'correlation': os.path.exists(os.path.join(PLOTS_FOLDER, 'correlation.png')),
        'scatter': os.path.exists(os.path.join(PLOTS_FOLDER, 'scatter.png')),
    }
    return render_template('visualize.html',
                           filename=current_filename,
                           columns=columns,
                           numeric_cols=numeric_cols,
                           plots_exist=plots_exist)

@app.route('/visualize/histogram', methods=['POST'])
def make_histogram():
    if current_df is None:
        flash('сначала загрузите данные', 'error')
        return redirect(url_for('index'))
    column = request.form.get('column')
    try:
        # берём не более 100 000 строк чтобы matplotlib не тормозил
        pdf = current_df.select(column).dropna().limit(100000).toPandas()
        plt.figure(figsize=(10, 6))
        plt.hist(pdf[column], bins=50, edgecolor='black', alpha=0.7, color='#2196F3')
        plt.title(f'histogram: {column}', fontsize=14)
        plt.xlabel(column)
        plt.ylabel('frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER, 'histogram.png'), dpi=100)
        plt.close()
        flash('гистограмма построена', 'success')
    except Exception as e:
        flash(f'ошибка: {str(e)}', 'error')
    return redirect(url_for('visualize'))

@app.route('/visualize/correlation', methods=['POST'])
def make_correlation():
    if current_df is None:
        flash('сначала загрузите данные', 'error')
        return redirect(url_for('index'))
    try:
        numeric_cols = get_numeric_cols(current_df)
        if len(numeric_cols) < 2:
            flash('недостаточно числовых столбцов для корреляции', 'error')
            return redirect(url_for('visualize'))
        df_clean = current_df.select(numeric_cols).dropna()
        # pyspark vectorassembler собирает столбцы в один вектор для correlation api
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol='corr_features')
        df_vector = assembler.transform(df_clean).select('corr_features')
        matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0].toArray()
        corr_df = pd.DataFrame(data=matrix, columns=numeric_cols, index=numeric_cols)
        fig_size = max(10, len(numeric_cols))
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(corr_df, annot=True, cmap='RdYlGn', center=0,
                    fmt='.2f', square=True, linewidths=0.5)
        plt.title('correlation matrix (pyspark)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER, 'correlation.png'), dpi=100)
        plt.close()
        flash('матрица корреляции построена', 'success')
    except Exception as e:
        flash(f'ошибка: {str(e)}', 'error')
    return redirect(url_for('visualize'))

@app.route('/visualize/scatter', methods=['POST'])
def make_scatter():
    if current_df is None:
        flash('сначала загрузите данные', 'error')
        return redirect(url_for('index'))
    col_x = request.form.get('col_x')
    col_y = request.form.get('col_y')
    try:
        # ограничиваем 50 000 точек для читаемости графика
        pdf = current_df.select(col_x, col_y).dropna().limit(50000).toPandas()
        plt.figure(figsize=(10, 6))
        plt.scatter(pdf[col_x], pdf[col_y], alpha=0.3, s=10, color='#FF5722')
        plt.title(f'{col_x} vs {col_y}', fontsize=14)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER, 'scatter.png'), dpi=100)
        plt.close()
        flash('scatter plot построен', 'success')
    except Exception as e:
        flash(f'ошибка: {str(e)}', 'error')
    return redirect(url_for('visualize'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if current_df is None:
        flash('сначала загрузите данные', 'error')
        return redirect(url_for('index'))

    columns = current_df.columns
    numeric_cols = get_numeric_cols(current_df)
    results = None

    if request.method == 'POST':
        target_col = request.form.get('target')
        feature_cols = request.form.getlist('features')
        model_type = request.form.get('model_type', 'random_forest')
        test_size = float(request.form.get('test_size', 0.2))

        if not target_col or not feature_cols:
            flash('выберите целевой столбец и хотя бы один признак', 'error')
            return redirect(url_for('train'))

        try:
            selected_cols = feature_cols + [target_col]
            sample_df = current_df.select(selected_cols).dropna()
            total = sample_df.count()

            # если строк больше 200 000 — берём подвыборку чтобы sklearn не тормозил
            if total > 200000:
                fraction = 200000 / total
                sample_df = sample_df.sample(fraction=fraction, seed=42)

            pdf = sample_df.toPandas()
            X = pdf[feature_cols]
            y = pdf[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model_map = {
                'random_forest': (
                    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    'random forest'
                ),
                'logistic_regression': (
                    LogisticRegression(max_iter=1000, random_state=42),
                    'logistic regression'
                ),
                'gradient_boosting': (
                    GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'gradient boosting'
                ),
            }

            model, model_name = model_map.get(model_type, model_map['random_forest'])

            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = round(time.time() - start_time, 2)

            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

            # сохраняем обученную модель на диск
            with open(os.path.join(MODELS_FOLDER, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        linewidths=0.5, linecolor='gray')
            plt.title(f'confusion matrix — {model_name}', fontsize=14)
            plt.ylabel('true label')
            plt.xlabel('predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER, 'confusion_matrix.png'), dpi=100)
            plt.close()

            has_importance = hasattr(model, 'feature_importances_')
            if has_importance:
                importance = model.feature_importances_
                fi_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importance
                }).sort_values('importance', ascending=True)
                plt.figure(figsize=(10, max(4, len(feature_cols) * 0.5)))
                plt.barh(fi_df['feature'], fi_df['importance'], color='#4CAF50')
                plt.title(f'feature importance — {model_name}', fontsize=14)
                plt.xlabel('importance')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_FOLDER, 'feature_importance.png'), dpi=100)
                plt.close()

            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'train_time': train_time,
                'train_size': len(X_train),
                'test_size_count': len(X_test),
                'total_rows': total,
                'has_importance': has_importance,
            }
            flash(f'модель обучена. accuracy: {accuracy}%', 'success')

        except Exception as e:
            flash(f'ошибка обучения: {str(e)}', 'error')

    return render_template('train.html',
                           filename=current_filename,
                           columns=columns,
                           numeric_cols=numeric_cols,
                           results=results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
