import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
from pathlib import Path

# Настройка
st.set_page_config(page_title="Предсказание стоимости автомобилей", page_icon="🚗")
st.title("Анализ данных автомобильных продаж")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LINEAR_MODEL_NAME = "Лучшая линейная регрессия"
CATBOOST_MODEL_NAME = "CatBoost"
MODEL_OPTIONS = [LINEAR_MODEL_NAME, CATBOOST_MODEL_NAME]
REQUIRED_INPUT_COLUMNS = [
    'name',
    'year',
    'km_driven',
    'fuel',
    'seller_type',
    'transmission',
    'owner',
    'mileage',
    'engine',
    'max_power',
    'seats',
]

CATBOOST_FEATURE_COLUMNS = [
    'name',
    'year',
    'km_driven',
    'fuel',
    'seller_type',
    'transmission',
    'owner',
    'mileage',
    'engine',
    'max_power',
    'seats',
    'brand',
    'car_age',
    'km_per_year',
    'power_per_engine',
    'engine_x_power',
]

@st.cache_data
def load_data():
    local_data_path = BASE_DIR / 'cars_train.csv'
    if local_data_path.exists():
        df = pd.read_csv(local_data_path)
    else:
        df = pd.read_csv('cars_train.csv')
    return df

def preprocess_data(df):
    """Очищает и преобразует данные для анализа"""
    df_processed = df.copy()
    df_processed['mileage'] = df_processed['mileage'].str.replace(' kmpl', '', regex=False)
    df_processed['mileage'] = df_processed['mileage'].str.replace(' km/kg', '', regex=False)
    df_processed['engine'] = df_processed['engine'].str.replace(' CC', '', regex=False)
    df_processed['max_power'] = df_processed['max_power'].str.replace(' bhp', '', regex=False)
    
    # Преобразуем к числам и обрабатываем NaN
    for col in ['mileage', 'engine', 'max_power']:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    return df_processed


def ensure_numpy_pickle_compatibility():
    import numpy.core.multiarray as multiarray
    import numpy.core.numeric as numeric
    import numpy.core.numerictypes as numerictypes
    import numpy.core.umath as umath

    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.numeric', numeric)
    sys.modules.setdefault('numpy._core.numerictypes', numerictypes)
    sys.modules.setdefault('numpy._core.multiarray', multiarray)
    sys.modules.setdefault('numpy._core.umath', umath)


@st.cache_resource
def load_artifacts(model_name):
    ensure_numpy_pickle_compatibility()

    if model_name == LINEAR_MODEL_NAME:
        model_path = MODEL_DIR / 'linear_regression_model.pkl'
        params_path = MODEL_DIR / 'linear_regression_params.pkl'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        params = {}
        if params_path.exists():
            with open(params_path, 'rb') as f:
                params = pickle.load(f)

        return model, params

    if model_name == CATBOOST_MODEL_NAME:
        model_path = MODEL_DIR / 'catboost_model.pkl'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model, {}

    raise ValueError(f"Неизвестная модель: {model_name}")


def add_feature_engineering(df):
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "В загруженном CSV не хватает обязательных признаков: "
            + ", ".join(missing_columns)
        )

    frame = preprocess_data(df)

    if 'torque' in frame.columns:
        frame = frame.drop(columns=['torque'])

    if 'name' in frame.columns:
        frame['brand'] = frame['name'].astype(str).str.split().str[0]

    frame['car_age'] = (2025 - frame['year']).clip(lower=1)
    frame['km_per_year'] = frame['km_driven'] / frame['car_age']
    frame['power_per_engine'] = frame['max_power'] / frame['engine'].replace(0, np.nan)
    frame['engine_x_power'] = frame['engine'] * frame['max_power']

    return frame


def prepare_linear_prediction_frame(df):
    frame = add_feature_engineering(df)
    return frame.drop(columns=['selling_price'], errors='ignore')


def prepare_catboost_prediction_frame(input_df):
    frame = add_feature_engineering(input_df)
    frame = frame.drop(columns=['selling_price'], errors='ignore')

    reference_frame = add_feature_engineering(df).drop(columns=['selling_price'], errors='ignore')

    missing_columns = [column for column in CATBOOST_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "В загруженном CSV не хватает признаков для CatBoost: "
            + ", ".join(missing_columns)
        )

    for column in CATBOOST_FEATURE_COLUMNS:
        if frame[column].dtype == 'object':
            frame[column] = frame[column].fillna('NA').astype(str)
        else:
            reference_values = reference_frame[column] if column in reference_frame.columns else frame[column]
            fill_value = pd.to_numeric(reference_values, errors='coerce').median()
            frame[column] = pd.to_numeric(frame[column], errors='coerce').fillna(fill_value)

    frame['seats'] = pd.to_numeric(frame['seats'], errors='coerce').fillna(reference_frame['seats'].median()).round().astype('Int64').astype(str)

    return frame[CATBOOST_FEATURE_COLUMNS]


def get_linear_feature_names(model, params):
    if hasattr(model, 'named_steps') and 'prep' in model.named_steps:
        preprocessor = model.named_steps['prep']
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    if params:
        for key in ('all_columns', 'num_columns', 'feature_names'):
            features = params.get(key)
            if features:
                return list(features)

    return []


def get_catboost_feature_names(model, frame):
    if hasattr(model, 'feature_names_') and model.feature_names_:
        return list(model.feature_names_)

    return list(frame.columns)

df = load_data()

# Навигация в сайдбаре
page = st.sidebar.radio(
    "Выберите раздел:",
    ["EDA", "Прогноз", "Значение весов"]
)

# --- РАЗДЕЛ EDA ---
if page == "EDA":
    st.header("EDA из проекта")
    
    st.write(f"**Данные:** {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Выбор визуализации
    viz_option = st.selectbox(
        "Выберите визуализацию:",
        ["Pairplot", "Heatmap корреляций"]
    )
    
    # 1. PAIRPLOT 
    if viz_option == "Pairplot":
        st.header("Pairplot данных")
        
        # Предобработка данных
        df_processed = preprocess_data(df)
        
        # Числовые признаки из задания 6
        numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
        # Удаляем строки с NaN в числовых признаках
        df_processed = df_processed[numerical_features].dropna()
        
        # Создаем pairplot
        st.write("**Pairplot числовых признаков:**")
        fig = sns.pairplot(df_processed)
        st.pyplot(fig)
        
        # Выводы из проекта
        st.info("""
        **Выводы по pairplot:**
        - selling_price растёт вместе с year (зависимость нелинейная)
        - selling_price уменьшается с увеличением km_driven
        - selling_price окололинейно зависит от max_power
        - При росте engine целевая переменная растёт до 2000, после остаётся на том же уровне
        """)
    
    # 2. HEATMAP корреляций (из задания 7 проекта)
    elif viz_option == "Heatmap корреляций":
        st.header("Heatmap по данным")
        # Предобработка данных
        df_processed = preprocess_data(df)
        
        # Числовые признаки
        numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
        # Удаляем строки с NaN в числовых признаках
        df_processed = df_processed[numerical_features].dropna()
        
        # Создаем heatmap
        st.write("**Корреляционная матрица (Пирсон):**")
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = df_processed.corr()
        sns.heatmap(corr_matrix, cmap="Blues", annot=True, fmt=".2f", ax=ax)
        ax.set_title('Корреляционная матрица данных')
        st.pyplot(fig)
        
        # Выводы из проекта
        st.info("""
        **Выводы по корреляционной матрице данных:**
        - Наиболее скоррелированы: max_power и selling_price (0.76)
        - Наименее скоррелированы: km_driven и max_power
        - Корреляция года и пробега: -0.37
        - Сильная положительная зависимость: engine и max_power (0.86)
        """)

# --- РАЗДЕЛ ПРОГНОЗ ---
elif page=="Прогноз":
    st.header("Прогнозирование стоимости")
    st.write("Загрузите CSV файл с данными автомобилей")

    selected_model = st.radio(
        "Выберите модель для прогноза:",
        MODEL_OPTIONS,
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Выберите CSV файл",
        type=["csv"],
        key="file_uploader",
    )

    if uploaded_file is not None:
        st.success("Файл загружен!")
        df_input = pd.read_csv(uploaded_file)
        st.write(f"Загружено {len(df_input)} записей")
        st.dataframe(df_input.head())

        if st.button("Сделать прогноз", type="primary"):
            try:
                model, params = load_artifacts(selected_model)
                st.success(f"Модель загружена: {selected_model}")

                with st.spinner("Выполняю прогноз..."):
                    if selected_model == LINEAR_MODEL_NAME:
                        df_pred = prepare_linear_prediction_frame(df_input)
                        predictions = np.asarray(model.predict(df_pred))
                    else:
                        df_pred = prepare_catboost_prediction_frame(df_input)
                        predictions = np.asarray(model.predict(df_pred))
                        predictions = np.expm1(predictions)
                        predictions = np.clip(predictions, 0, None)

                st.subheader("Результаты:")
                results_df = pd.DataFrame({
                    '№': range(1, len(predictions) + 1),
                    'Предсказанная цена': predictions,
                })
                st.dataframe(results_df)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Средняя", f"{predictions.mean():,.0f}")
                with col2:
                    st.metric("Минимум", f"{predictions.min():,.0f}")
                with col3:
                    st.metric("Максимум", f"{predictions.max():,.0f}")

            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.write("Убедитесь, что файл содержит нужные признаки и что артефакты модели доступны.")
    else:
        st.info("Выберите CSV файл для загрузки")

        with st.expander("Какой формат данных нужен?"):
            st.write(
                """
                Файл должен содержать исходные признаки автомобиля без целевой переменной `selling_price`.

                Для обеих моделей приложение автоматически добавит нужную предобработку и feature engineering.
                Минимальный набор столбцов:
                - name (название модели)
                - year (год)
                - km_driven (пробег)
                - mileage (расход)
                - engine (объем)
                - max_power (мощность)
                - seats (места)
                - fuel (топливо)
                - seller_type (продавец)
                - transmission (коробка)
                - owner (владелец)
                """
            )


#-----ВЕСА МОДЕЛИ

elif page == 'Значение весов':
    st.header("Веса модели")

    selected_model = st.radio(
        "Выберите модель:",
        MODEL_OPTIONS,
        horizontal=True,
    )

    try:
        model, params = load_artifacts(selected_model)

        if selected_model == LINEAR_MODEL_NAME:
            if hasattr(model, 'named_steps') and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'coef_'):
                st.success("Модель загружена")

                coefficients = model.named_steps['model'].coef_
                feature_names = get_linear_feature_names(model, params)

                if not feature_names or len(feature_names) != len(coefficients):
                    feature_names = [f'Признак_{i}' for i in range(len(coefficients))]

                coef_df = pd.DataFrame({
                    'Признак': feature_names,
                    'Коэффициент': coefficients,
                    'Абсолютное значение': np.abs(coefficients),
                }).sort_values('Абсолютное значение', ascending=False)

                st.subheader("Таблица коэффициентов")
                st.dataframe(coef_df.style.format({'Коэффициент': '{:.6f}', 'Абсолютное значение': '{:.6f}'}))

                st.subheader("Распределение всех коэффициентов")
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                colors = ['red' if x < 0 else 'green' for x in coefficients]
                sizes = np.abs(coefficients) * 100 / np.max(np.abs(coefficients))

                ax2.scatter(range(len(coefficients)), coefficients, c=colors, s=sizes, alpha=0.6)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_xlabel('Индекс признака')
                ax2.set_ylabel('Значение коэффициента')
                ax2.set_title('Распределение коэффициентов модели')
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig2)

                st.subheader("Ключевые выводы")

                top_positive = coef_df[coef_df['Коэффициент'] > 0].head(3)
                top_negative = coef_df[coef_df['Коэффициент'] < 0].head(3)

                if len(top_positive) > 0:
                    st.write("**Самые важные положительные признаки:**")
                    for _, row in top_positive.iterrows():
                        st.write(f"- {row['Признак']}: {row['Коэффициент']:.2f}")

                if len(top_negative) > 0:
                    st.write("**Самые важные отрицательные признаки:**")
                    for _, row in top_negative.iterrows():
                        st.write(f"- {row['Признак']}: {row['Коэффициент']:.2f}")
            else:
                st.error("У линейной модели нет доступных коэффициентов")

        else:
            if hasattr(model, 'get_feature_importance'):
                st.success("CatBoost модель загружена")

                catboost_frame = prepare_catboost_prediction_frame(df)
                feature_names = get_catboost_feature_names(model, catboost_frame)
                importances = model.get_feature_importance()

                if len(feature_names) != len(importances):
                    feature_names = [f'Признак_{i}' for i in range(len(importances))]

                importance_df = pd.DataFrame({
                    'Признак': feature_names,
                    'Важность': importances,
                }).sort_values('Важность', ascending=False)

                st.subheader("Таблица важности признаков")
                st.dataframe(importance_df)

                st.subheader("Топ-10 признаков")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                top_importance = importance_df.head(10).iloc[::-1]
                ax2.barh(top_importance['Признак'], top_importance['Важность'], color='steelblue')
                ax2.set_xlabel('Важность')
                ax2.set_ylabel('Признак')
                ax2.set_title('Топ-10 признаков CatBoost')
                ax2.grid(True, axis='x', alpha=0.3)

                st.pyplot(fig2)

                st.subheader("Ключевые выводы")
                for _, row in importance_df.head(3).iterrows():
                    st.write(f"- {row['Признак']}: {row['Важность']:.2f}")
            else:
                st.error("У CatBoost модели нет функции важности признаков")

    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        import traceback
        st.write("Полная ошибка:")
        st.code(traceback.format_exc())