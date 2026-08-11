import requests
r = requests.post('http://127.0.0.1:8000/analyze-stress?session_id=tmp_test')
print(r.status_code)
try:
    print(r.json())
except Exception:
    print(r.text)
