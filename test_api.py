import requests

url = "http://localhost:8000/api/screen-cvs"

files = {
    "files": open("../test_cvs/my_cv.pdf", "rb")
}

data = {
    "criteria": '{"required_skills": ["Python", "SQL"], "education": "Bachelor\'s in CS", "selection_threshold": 70}'
}

response = requests.post(url, files=files, data=data)

print(response.status_code)
print(response.json())
