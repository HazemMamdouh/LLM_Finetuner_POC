from starlette.testclient import TestClient
from app.main import app
import json

client = TestClient(app)


def test_alive():
    response = client.get('/alive')
    assert response.status_code == 200
    assert response.json() == {'message': 'Server is Alive'}

