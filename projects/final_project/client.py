import json
import requests

# Формируем сырой тестовый запрос по нашему дому в Неваде (NV)
# Сайт отправляет ровно те 36 колонок, которые ожидает наш пайплайн
house_data = {
    "city": "Las Vegas",
    "sqft": 2863.0,
    "zipcode": "89101",
    "state": "NV",
    "Year built": 2005.0,
    "Remodeled year": 0.0,
    "lotsize": 6534.0,
    "heating_clean": "forced_air",
    "cooling_clean": "central_ac",
    "parking_clean": "garage",
    "dist_elementary": 0.8,
    "dist_middle": 1.2,
    "dist_high": 1.5,
    "school_max_rating": 7.0,
    "school_mean_rating": 5.3,
    "school_median_rating": 5.0,
    "is_pool": 0,
    "is_sqft_unknown": 0,
    "is_start_price": 0,
    "has_baths_mention": 0,
    "has_appliances": 1,
    "has_extra_rooms": 0,
    "beds_clean": "3",
    "baths_clean": "2.5",
    "land_scale": "under_acre",
    "is_historic": 0,
    "is_modern": 1,
    "is_luxury": 0,
    "is_waterfront_exotic": 0,
    "property_type_clean": "house",
    "stories_clean": "1.5_to_2",
    "has_fireplace": 1,
    "has_luxury_finishing": 0,
    "has_outdoor_amenities": 1,
    "status_clean": "unknown",
    "is_remodeled": 0,
}

# Указываем точный локальный адрес нашего запущенного Flask-сервера
url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

print("Отправка тестового POST-запроса на Flask-сервер...")

try:
    # Отправляем JSON-данные на сервер
    response = requests.post(url, data=json.dumps(house_data), headers=headers)

    # Выводим ответ
    print(f"Статус-код ответа сервера: {response.status_code}")
    if response.status_code == 200:
        print("\nУспешный ответ от продакшен-API:")
        res_json = response.json()
        print(f" - Рассчитанная стоимость объекта: "
              f"${res_json['predicted_price']:,.2f}")
        print(f" - Статус транзакции: {res_json['status']}")
    else:
        print(f"Ошибка сервера: {response.text}")

except Exception as e:
    print(f"Не удалось связаться с сервером: {e}")