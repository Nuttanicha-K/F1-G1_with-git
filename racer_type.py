#%%
"""ver 1 : ทำผิดดันใช้ความเร็วสูงสุดมาเฉลี่ย และทำแค่ปี2023"""
import fastf1
from fastf1 import plotting
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plotting.setup_mpl(misc_mpl_mods=False)

#เปิดระบบ cache
fastf1.Cache.enable_cache('cache')  # แก้ path ตามที่ตั้งไว้
#ดึงข้อมูลจากทุกสนามของปี 2023
year = 2023
schedule = fastf1.get_event_schedule(year)
#เก็บข้อมูลนักแข่งที่สนามไม่มีฝนและแข่งจบ
driver_stats = {}
for _, row in schedule.iterrows():#for index, row in ... ในกรณีนี้_เพราะไม่ใช้ ไม่สน no care!
    #ข้ามsprint
    if row['EventFormat'] != 'conventional':  
        continue
    # ดึงข้อมูลของแต่ละสนาม (แต่ละแถว)
    rnd = row['RoundNumber'] #เลขสนาม
    event_name = row['EventName'] #ชื่อสนาม
    
    try:
        session = fastf1.get_session(year, rnd, 'R')
        session.load()

        # ข้ามสนามที่มีฝนตก
        weather = session.weather_data
        if weather['Rainfall'].sum() > 0:
            continue

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            
            if laps.empty or drv not in session.results.index:
                continue
            
            #ดึงแถวผลการแข่งขันของนักแข่งคนนั้นจากตาราง session.results จะได้ข้อมูลเช่น Grid Position, Finish Position, Status ฯลฯ
            result = session.results.loc[drv]
            if result['Status'] != 'Finished':#เอานักแข่งที่แข่งจบ
                continue

            max_speed = laps['SpeedST'].max() #SpeedST คือความเร็วในเส้นตรง
            avg_speed = laps['SpeedST'].mean()
            grid_pos = result['GridPosition']
            finish_pos = result['Position']

            #"เก็บสถิติแต่ละสนามรวมไว้ต่อคน" 
            if drv not in driver_stats: # ถ้ายังไม่เคยเก็บข้อมูลของนักแข่งคนนี้ (ยังไม่มี key 'VER', 'HAM', ...)
                driver_stats[drv] = {
                    'Name': result['FullName'],
                    'MaxSpeeds': [],
                    'AvgSpeeds': [],
                    'GridPositions': [],
                    'FinishPositions': []
                }

            driver_stats[drv]['MaxSpeeds'].append(max_speed)
            driver_stats[drv]['AvgSpeeds'].append(avg_speed)
            driver_stats[drv]['GridPositions'].append(grid_pos)
            driver_stats[drv]['FinishPositions'].append(finish_pos)
    
    except Exception as e:
        print(f"❌ Error loading {event_name}: {e}")

# สร้าง DataFrame สำหรับการ clustering #รวมค่าเฉลี่ยของนักแข่งแต่ละคน
data = []
names = []
for drv, stats in driver_stats.items(): #stats คือข้อมูลของนักแข่งแต่ละคน (dict ข้างใน)
    names.append(stats['Name'])
    data.append([
        np.mean(stats['MaxSpeeds']),
        np.mean(stats['AvgSpeeds']),
        np.mean(stats['GridPositions']),
        np.mean(stats['FinishPositions']),
    ])

# ขั้นตอนรวมข้อมูลทั้งหมดเข้าเป็นตาราง DataFrame
df = pd.DataFrame(data, columns=['MaxSpeed', 'AvgSpeed', 'GridPos', 'FinishPos'])
df['Name'] = names #เพิ่มคอลัมน์ใหม่ชื่อ "Name" เพื่อเก็บชื่อจริงของนักแข่ง

# ทำ KMeans Clustering
X = df[['MaxSpeed', 'AvgSpeed', 'GridPos', 'FinishPos']]
kmeans = KMeans(n_clusters=3, random_state=42) #ตั้งค่าเมล็ดสุ่ม (random seed) 42 is the answer to life, the universe, and everything” #random_state=42 → ใช้สำหรับทำให้ผลการแบ่งกลุ่ม เหมือนเดิมทุกครั้
df['Cluster'] = kmeans.fit_predict(X) #เพิ่มคอลัมน์ "Cluster" ลงใน df เพื่อเก็บผลลัพธ์ของการจัดกลุ่ม

# แสดงตาราง 
print(df[['Name', 'Cluster']].sort_values(by='Cluster'))

# วาดกราฟด้วย PCA ลดมิติ
pca = PCA(n_components=2) #PCA (Principal Component Analysis) → ใช้ลดจาก 4 มิติ (MaxSpeed, AvgSpeed, ...) เหลือ 2 มิติ
components = pca.fit_transform(X) #components จะเป็น array ขนาด (จำนวนนักแข่ง, 2)
#วาดกราฟ
plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique(): #วนลูป “ค่าที่ไม่ซ้ำกัน” (unique) ของคอลัมน์ Cluster
    idx = df['Cluster'] == cluster
    plt.scatter(components[idx, 0], components[idx, 1], label=f'Cluster {cluster}') #พิกัด X-Y ของนักแข่งแต่ละคนใน cluster นั้น #components คือข้อมูลที่ถูกลดมิติด้วย PCA (2 มิติ) , components[idx, 0] → ค่าแกน X ของนักแข่งที่อยู่ใน cluster นั้น , components[idx, 1] → ค่าแกน Y ของนักแข่งที่อยู่ใน cluster นั้น

for i, name in enumerate(df['Name']):
    plt.text(components[i, 0], components[i, 1], name, fontsize=8) #components[i, 0], components[i, 1] → ตำแหน่งของแต่ละคนในกราฟ

plt.title('F1 2023 Driver Clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
"""verนี้รอพรุ่งนี้chatหมดโค้วต้าละ"""
"""ver 2 : ช่วยแบ่งกลุ่มในสนามrace ที่ฝนไม่ตกและมีการแข่งจนจบ โดยที่ใช้คุณสมบัติดังนี้ 1.ความเร็วสูงสุดในแต่ละrace 2.ความเร็วของทุกraceเฉลี่ย 3.ค่าเฉลี่ยpositionเริ่มต้น 4.ค่าเฉลี่ยpositionสุดท้าย ตั้งแต่ปี2021จนถึง2024 ได้มั้ย และมีคำถามอีก1คือหากใช้คุณสมบัติที่ว่ามาดังนั้นนักแข่งคนหนึ่งสามารถอยู่ได้หลายกลุ่มใช่มั้ย เนื่องจากค่าต่าง ๆ ในแต่ละraceต่างกัน เช่น ความเร็วสูงสุดใยแต่ละrace"""

