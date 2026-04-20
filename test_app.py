from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def sample_request_body():
    return {
        "passenger_count": 1.0,
        "trip_distance": 3.2,
        "fare_amount": 18.5,
        "pickup_hour": 14,
        "pickup_day_of_week": 2,
        "is_weekend": 0,
        "trip_duration_minutes": 16.0,
        "trip_speed_mph": 12.0,
        "log_trip_distance": 1.435,
        "fare_per_mile": 5.78,
        "fare_per_minute": 1.16,
        "pickup_borough": "Manhattan",
        "dropoff_borough": "Queens",
    }


def test_single_prediction_success():
    payload = sample_request_body()
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'prediction_id' in data
    assert data['predicted_tip_amount'] >= 0


def test_batch_prediction_success():
    payload = sample_request_body()
    response = client.post('/predict/batch', json={'records': [payload, payload]})
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 2
    assert len(data['predictions']) == 2


def test_invalid_input_rejected():
    payload = sample_request_body()
    payload['pickup_hour'] = 30
    response = client.post('/predict', json=payload)
    assert response.status_code == 422


def test_health_check():
    response = client.get('/health')
    assert response.status_code == 200
    assert 'model_loaded' in response.json()


def test_zero_distance_rejected():
    payload = sample_request_body()
    payload['trip_distance'] = 0
    response = client.post('/predict', json=payload)
    assert response.status_code == 422


def test_model_info():
    response = client.get('/model/info')
    assert response.status_code == 200
    data = response.json()
    assert 'feature_names' in data
    assert 'training_metrics' in data
