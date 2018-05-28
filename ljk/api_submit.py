
# coding: utf-8

import requests
import json
import datetime
# ans_files = ['ans_l24.csv', 'ans_l12.csv']
# des = ['arma-12.csv', 'forecast-12.csv']
# ans_files = ['ans-better.csv']
# des = ['merged']
ans_files = ['ans_l12.csv']
des = ['forecast-12-s18.csv']
for i in range(1):
    s = ans_files[i]
    de = des[i]
    print('submitting '+s)
    for _ in range(10):
        now = datetime.datetime.now()
        print(now)
        if now.hour >= 8 or now.hour == 7 and now.minute >= 58:
            print('TLE')
            break
        files={'files': open(s,'rb')}

        data = {
            "user_id": "lazycal",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
            "team_token": "4157c157c8f8a7ddd9420a7e8f19baa3da31e652da11bc0553eb121a2949cf69", #your team_token.
            "description": de,  #no more than 40 chars.
            "filename": s, #your filename
        }
        url = 'https://biendata.com/competition/kdd_2018_submit/'
        print(data)
        response = requests.post(url, files=files, data=data)
        print(response.text)
        if response.text.find('"success": true') != -1: break



