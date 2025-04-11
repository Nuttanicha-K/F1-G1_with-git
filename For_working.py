#%%
import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
import pandas as pd
import fastf1.plotting
import numpy as np
import sklearn
from sklearn.cluster import KMeans
#%%
#ขอลองเอาดาต้า1แมตช์มาทดลองทำเป็นโมเดลเล็ก ๆ ก่อนนะแม่
session = fastf1.get_session(2023, 'Australian', 'R')
session.load()
track_status = session.track_status
print(track_status)
# %%
#อันนี้ลองextractเอาแต่timeที่มีyellow flag กับ safety carโผล่มา
df_track_status = track_status

yellow_flag_safety_appeared = df_track_status[df_track_status['Status'].isin(['2', '4'])] #สเตตัสเป็นstringงับ
print(yellow_flag_safety_appeared)

time_wheren_yellow_flag_safety_appeared = yellow_flag_safety_appeared['Time']
print(time_wheren_yellow_flag_safety_appeared)
# %%
#อันนี้จะลองดึงข้อมูลความเร็วรถนักแข่งออกมาเป็นตาราง
fast_ham = session.laps.pick_drivers('HAM').pick_fastest()
ham_car_data = fast_ham.get_car_data()
t2 = ham_car_data['Time']
v2 = ham_car_data['Speed']

timeoverspeed_ham = pd.DataFrame({
    'Time': ham_car_data['Time'],
    'Speed': ham_car_data['Speed']
})

print(timeoverspeed_ham.to_string()) 
# %%
# ทำให้ข้อมูลเวลาธงเหลืองกับความเร็วรถนักแข่งทั้งสองเปนสตริงก่อนละกันแม่
timeoverspeed_ham['Time_str'] = timeoverspeed_ham['Time'].astype(str)
time_wheren_yellow_flag_safety_appeared_str = time_wheren_yellow_flag_safety_appeared.astype(str)

# %%
#พยายามแมตช์เวลาที่ธงเหลืองกับsafetycarโผล่มา กับ speed รถของนักแข่ง ณ เวลานั้น
print(timeoverspeed_ham['Time'].min(), timeoverspeed_ham['Time'].max())
print(time_wheren_yellow_flag_safety_appeared.min(), time_wheren_yellow_flag_safety_appeared.max())
def find_closest_time(yellow_time, available_times):
    closest_time = available_times.iloc[(available_times - yellow_time).abs().argmin()]
    return closest_time

results = []

for yellow_time in time_wheren_yellow_flag_safety_appeared:
    closest_time = find_closest_time(yellow_time, timeoverspeed_ham['Time'])
    
    
    corresponding_speed = timeoverspeed_ham[timeoverspeed_ham['Time'] == closest_time]['Speed'].values[0]
    
    results.append({
        'Yellow Flag Time': yellow_time,
        'Closest Time': closest_time,
        'Speed': corresponding_speed
    })
matched_df = pd.DataFrame(results)
print(matched_df)

#ปัญหาของตอนนี้คือไม่สามารถmatchเวลาจากความเร็วรถของนักแข่งเทียบกับเวลาของธงได้ 
# คือมันmatchได้ แต่มันไม่accurateขนาดนั้น น่าจะเขียนโค้ดผิดหรือมันผิด ฉันไม่ผิด

#%%
#เดี๋ยวจะลองmatchกับสภาพอากาศดู
session = fastf1.get_session(2023, 'Australian', 'R')
session.load()
weather_data = session.weather_data
print(weather_data)

#%%
df_track_status = track_status

yellow_flag_safety_appeared = df_track_status[df_track_status['Status'].isin(['2', '4'])] #สเตตัสเป็นstringงับ
print(yellow_flag_safety_appeared)

def find_closest_time(yellow_time, available_times):
    time_diff = abs(available_times - yellow_time)
    closest_idx = time_diff.idxmin()
    
    return available_times.loc[closest_idx], weather_data.loc[closest_idx, 'AirTemp']


matched = []

for yellow_time in time_wheren_yellow_flag_safety_appeared:
    closest_time, speed = find_closest_time(yellow_time, weather_data['Time'])
    matched.append({
        'Yellow Flag Time': yellow_time,
        'Closest Time': closest_time,
        'AirTemp': speed    #ฉันก็งงคือกันทำไมใส่airtempไม่ติด แตใส speeed ละขึ้นข้อมูล air tempให้
    })

matched_df = pd.DataFrame(matched)
print(matched_df)
