import requests
from requests.api import get

def get_translate(text):
    client_id = "TdvMvERiiQk5a4wTgdE6" # <-- client_id 기입
    client_secret = "LLFGSXPgyt" # <-- client_secret 기입

    data = {'text' : text,
            'source' : 'en',
            'target': 'ko'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if(rescode==200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:" , rescode)

