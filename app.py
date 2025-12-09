import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Настройка
st.set_page_config(page_title="Предсказание стоимости автомобилей", page_icon="🚗")
st.title("Анализ данных автомобильных продаж")

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    return df
df = load_data()

# Навигация в сайдбаре
page = st.sidebar.radio(
    "Выберите раздел:",
    ["EDA", "Прогноз", "Значение весов"]
)

# --- РАЗДЕЛ EDA ---
if page == "EDA":
    st.header("EDA из проекта")
    
    # Загрузка данных
    @st.cache_data
    def load_data():
        df = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
        return df
    
    df = load_data()
    
    st.write(f"**Данные:** {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Выбор визуализации
    viz_option = st.selectbox(
        "Выберите визуализацию:",
        ["Pairplot", "Heatmap корреляций"]
    )
    
    # 1. PAIRPLOT 
    if viz_option == "Pairplot":
        st.header("Pairplot данных")
        
        # Предобработка как в проекте
        df_processed = df.copy()
        df_processed['mileage'] = df_processed['mileage'].str.replace(' kmpl', '', regex=False)
        df_processed['mileage'] = df_processed['mileage'].str.replace(' km/kg', '', regex=False)
        df_processed['engine'] = df_processed['engine'].str.replace(' CC', '', regex=False)
        df_processed['max_power'] = df_processed['max_power'].str.replace(' bhp', '', regex=False)
        
        # Преобразуем к числам
        for col in ['mileage', 'engine', 'max_power']:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Числовые признаки из задания 6
        numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
        
        # Создаем pairplot
        st.write("**Pairplot числовых признаков:**")
        fig = sns.pairplot(df_processed[numerical_features])
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
        # Та же предобработка
        df_processed = df.copy()
        df_processed['mileage'] = df_processed['mileage'].str.replace(' kmpl', '', regex=False)
        df_processed['mileage'] = df_processed['mileage'].str.replace(' km/kg', '', regex=False)
        df_processed['engine'] = df_processed['engine'].str.replace(' CC', '', regex=False)
        df_processed['max_power'] = df_processed['max_power'].str.replace(' bhp', '', regex=False)
        
        for col in ['mileage', 'engine', 'max_power']:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Числовые признаки
        numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
        
        # Создаем heatmap
        st.write("**Корреляционная матрица (Пирсон):**")
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = df_processed[numerical_features].corr()
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
    
    # 1. Показываем инструкцию
    st.write("Загрузите CSV файл с данными автомобилей")
    
    # 2. Кнопка загрузки файла
    uploaded_file = st.file_uploader(
        "Выберите CSV файл",
        type=["csv"],
        key="file_uploader"  # Уникальный ключ
    )
    
    # 3. Если файл загружен
    if uploaded_file is not None:
        st.success("Файл загружен!")
        # Читаем файл
        df = pd.read_csv(uploaded_file)
        # Показываем что загрузилось
        st.write(f"Загружено {len(df)} записей")
        st.dataframe(df.head())
        
        # 4. Кнопка для прогноза
        if st.button("Сделать прогноз", type="primary"):
            try:
                # Загружаем модель
                with open('models/linear_regression_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                st.success("Модель загружена")
                # Делаем прогноз
                with st.spinner("Выполняю прогноз..."):
                    predictions = model.predict(df)
                # Показываем результаты
                st.subheader("Результаты:")
                results_df = pd.DataFrame({
                    '№': range(1, len(predictions) + 1),
                    'Предсказанная цена': predictions
                })
                st.dataframe(results_df)
                # Статистика
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Средняя", f"{predictions.mean():,.0f}")
                with col2:
                    st.metric("Минимум", f"{predictions.min():,.0f}")
                with col3:
                    st.metric("Максимум", f"{predictions.max():,.0f}")
                    
            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.write("Убедитесь, что файлы .pkl находятся в той же папке")
    
    # 5. Если файл еще не загружен
    else:
        st.info("Выберите CSV файл для загрузки")
        
        # Пример формата файла
        with st.expander("Какой формат данных нужен?"):
            st.write("""
            Файл должен содержать те же признаки, что и при обучении модели:
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
            """)


#-----ВЕСА МОДЕЛИ

elif page == 'Значение весов':
    st.header("Веса модели")
    
    # Загружаем модель
    try:
        with open('models/linear_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Проверяем, есть ли у модели коэффициенты
        if hasattr(model, 'coef_'):
            st.success("Модель загружена")
            
            # 1. Получаем коэффициенты
            coefficients = model.coef_
            
            # 2. Получаем названия признаков (если есть в params)
            feature_names = [] 
            try:
                with open('models/linear_regression_params.pkl', 'rb') as f:
                    params = pickle.load(f)
                    feature_names = params.get('feature_names', [])
            except Exception as e:
                st.warning(f"Не удалось загрузить feature_names: {e}")
            
            # Если feature_names пуст или не совпадает по длине, создаем свои
            if not feature_names or len(feature_names) != len(coefficients):
                feature_names = [f'Признак_{i}' for i in range(len(coefficients))]
            
            # 3. Создаем DataFrame
            coef_df = pd.DataFrame({
                'Признак': feature_names,
                'Коэффициент': coefficients,
                'Абсолютное значение': np.abs(coefficients)
            }).sort_values('Абсолютное значение', ascending=False)
            
            # 4. Показываем таблицу
            st.subheader("Таблица коэффициентов")
            st.dataframe(coef_df.style.format({'Коэффициент': '{:.6f}', 'Абсолютное значение': '{:.6f}'}))
            
            # 5. Визуализация
            st.subheader("Распределение всех коэффициентов")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Разные цвета и размеры
            colors = ['red' if x < 0 else 'green' for x in coefficients]
            sizes = np.abs(coefficients) * 100 / np.max(np.abs(coefficients))
            
            ax2.scatter(range(len(coefficients)), coefficients, 
                       c=colors, s=sizes, alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Индекс признака')
            ax2.set_ylabel('Значение коэффициента')
            ax2.set_title('Распределение коэффициентов модели')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
            
            # 6. Самые важные признаки
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
            st.error(" У модели нет атрибута coef_")
            
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        import traceback
        st.write("Полная ошибка:")
        st.code(traceback.format_exc())