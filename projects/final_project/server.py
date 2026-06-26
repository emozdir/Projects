import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Инициализируем веб-приложение Flask
app = Flask(__name__)


# Дублирую классы
class EffectiveAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2026):
        self.current_year = current_year
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        if 'Remodeled year' in X_out.columns:
            X_out['effective_age'] = (
                (self.current_year - X_out['Remodeled year']).where(
                    X_out['Remodeled year'] > 0, -1
                ).astype(float)
            )
        else:
            X_out['effective_age'] = np.full(len(X_out), -1.0)
        return X_out


class GeographicCoordinatesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, zips_df):
        self.zips_df = zips_df
        self.city_lat_, self.city_lon_ = None, None
        self.state_lat_, self.state_lon_ = None, None
        self.global_lat_, self.global_lon_ = None, None
        
    def fit(self, X, y=None):
        X_merged = X.merge(
            self.zips_df, left_on='zipcode', right_on='zip', how='left'
        )
        self.city_lat_ = X_merged.groupby('city')['latitude'].mean()
        self.city_lon_ = X_merged.groupby('city')['longitude'].mean()
        self.state_lat_ = X_merged.groupby('state')['latitude'].mean()
        self.state_lon_ = X_merged.groupby('state')['longitude'].mean()
        self.global_lat_ = X_merged['latitude'].mean()
        self.global_lon_ = X_merged['longitude'].mean()
        return self
    
    def transform(self, X):
        X_out = X.copy()
        X_out = X_out.merge(
            self.zips_df, left_on='zipcode', right_on='zip', how='left'
        ).set_index(X.index)
        X_out['latitude'] = X_out['latitude'].fillna(
            X_out['city'].map(self.city_lat_)
        )
        X_out['longitude'] = X_out['longitude'].fillna(
            X_out['city'].map(self.city_lon_)
        )
        X_out['latitude'] = X_out['latitude'].fillna(
            X_out['state'].map(self.state_lat_)
        )
        X_out['longitude'] = X_out['longitude'].fillna(
            X_out['state'].map(self.state_lon_)
        )
        X_out['latitude'] = X_out['latitude'].fillna(self.global_lat_)
        X_out['longitude'] = X_out['longitude'].fillna(self.global_lon_)
        return X_out


class BedsBathsCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, beds_clip=5, baths_clip=4):
        self.beds_clip = beds_clip
        self.baths_clip = baths_clip
        self.beds_median_, self.baths_median_ = None, None
        
    def fit(self, X, y=None):
        if 'beds_clean' in X.columns:
            self.beds_median_ = pd.to_numeric(
                X['beds_clean'].replace('unknown', np.nan), errors='coerce'
            ).median()
        else:
            self.beds_median_ = 3.0
        if 'baths_clean' in X.columns:
            self.baths_median_ = pd.to_numeric(
                X['baths_clean'].replace('unknown', np.nan), errors='coerce'
            ).median()
        else:
            self.baths_median_ = 2.0
        return self
    
    def transform(self, X):
        X_out = X.copy()
        if 'beds_clean' in X_out.columns:
            X_out['is_beds_unknown'] = (
                X_out['beds_clean'] == 'unknown'
            ).astype(int)
            X_out['beds_clean'] = X_out['beds_clean'].replace(
                'unknown', self.beds_median_
            ).astype(float)
            X_out['beds_clean'] = np.round(
                X_out['beds_clean']
            ).clip(upper=self.beds_clip).astype(int)
        if 'baths_clean' in X_out.columns:
            X_out['is_baths_unknown'] = (
                X_out['baths_clean'] == 'unknown'
            ).astype(int)
            X_out['baths_clean'] = X_out['baths_clean'].replace(
                'unknown', self.baths_median_
            ).astype(float)
            X_out['baths_clean'] = np.round(
                X_out['baths_clean']
            ).clip(upper=self.baths_clip).astype(int)
        return X_out


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['sqft', 'lotsize'], quantile=0.99):
        self.columns = columns
        self.quantile = quantile
        self.upper_bounds_ = {}
        
    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                self.upper_bounds_[col] = X[col].quantile(self.quantile)
        return self
    
    def transform(self, X):
        X_out = X.copy()
        for col in self.columns:
            if col in X_out.columns and col in self.upper_bounds_:
                X_out[col] = X_out[col].clip(upper=self.upper_bounds_[col])
        return X_out


# 2. Загружаем наш готовый обученный пайплайн XGBoost из файла
# Он автоматически подтянет все кастомные трансформеры и веса модели
try:
    # Автоматически находим точную папку, где лежит этот скрипт server.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Намертво склеиваем её с подпапкой data и файлом модели
    model_path = os.path.join(base_dir, "data", "final_xgb_model.joblib")
    model_pipeline = joblib.load(model_path)
    print("Продакшен-модель XGBoost успешно загружена в память сервера!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model_pipeline = None


# 3. Создаем веб-маршрут (эндпоинт) для обработки входящих POST-запросов
@app.route("/predict", methods=["POST"])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Модель не загружена на сервере"}), 500

    try:
        json_data = request.get_json()
        input_df = pd.DataFrame([json_data])

        for col in input_df.columns:
            # Исключаем почтовые индексы и города из перевода в числа
            if col in ["zipcode", "city"]:
                input_df[col] = input_df[col].astype(str)
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors="ignore")

        # Отправляем восстановленный DataFrame в пайплайн
        predicted_price = model_pipeline.predict(input_df)
        final_price = float(np.round(predicted_price, 2))

        return jsonify(
            {"predicted_price": final_price, "status": "success"}
        ), 200

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400


# 4. Запускаем сервер на локальном хосте (порт 5000)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)