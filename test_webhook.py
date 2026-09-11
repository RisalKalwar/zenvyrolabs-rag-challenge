import requests

url = "https://halarkalwar.app.n8n.cloud/webhook/cv-screening"  # paste your actual test URL here

payload = {
    "candidates": [
        {
            "candidate_name": "Test Selected",
            "email": "kalwar.risal@gmail.com",  # your real address, not the placeholder,
            "email_flagged": False,
            "score": 85,
            "category": "Selected",
            "reasoning": "Strong match",
            "feedback": "Great experience with Python and SQL."
        }
    ]
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.text)