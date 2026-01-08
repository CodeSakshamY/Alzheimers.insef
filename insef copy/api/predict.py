# api/predict.py
from predict_logic import predict_single

def handler(request, response):
    try:
        data = request.json
        result = predict_single(data)
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        response.send(result)
    except Exception as e:
        response.status_code = 500
        response.headers['Content-Type'] = 'application/json'
        response.send({'error': str(e)})
