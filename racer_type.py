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
"""ver 2 : race ที่ฝนไม่ตกและมีการแข่งจนจบ **นักแข่งคนหนึ่งอยู่ได้หลายกลุ่ม"""
"""
ความเร็วสูงสุดของนักแข่งในสนามนั้น
ความเร็วเฉลี่ยของนักแข่งในสนามนั้น
ตำแหน่งสตาร์ทของนักแข่งในสนามนั้น
ตำแหน่งเข้าเส้นชัยของนักแข่งในสนามนั้น
"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
fastf1.Cache.enable_cache('cache')  # เปิดแคช

data = []
names = []

for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year)
    
    for _, row in schedule.iterrows():
        if row['EventFormat'] != 'conventional':
            continue  # ข้าม Sprint

        session = fastf1.get_session(year, row['RoundNumber'], 'R')
        try:
            session.load()
        except:
            continue

        weather = session.weather_data
        if weather['Rainfall'].sum() > 0:
            continue  # ข้ามสนามที่ฝนตก

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            max_speed = laps['SpeedST'].max()
            avg_speed = laps['SpeedST'].mean()
            grid_pos = result['GridPosition']
            finish_pos = result['Position']

            names.append(f"{year} {row['EventName']} - {session.get_driver(drv)['FullName']}")
            data.append([max_speed, avg_speed, grid_pos, finish_pos])

# --- ทำ DataFrame ---
df = pd.DataFrame(data, columns=['MaxSpeed', 'AvgSpeed', 'GridPos', 'FinishPos'])
df['Name'] = names

# --- Clustering ---
X = df[['MaxSpeed', 'AvgSpeed', 'GridPos', 'FinishPos']]
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- ตั้งชื่อคลัสเตอร์ (สามารถปรับชื่อให้เหมาะกับผลลัพธ์จริงได้) ---
df['ClusterLabel'] = df['Cluster'].map({
    0: 'Maprang',
    1: 'Sense',
    2: 'Pooh',
    3: 'Fern'
})

# --- แสดงผลแบบกราฟ (PCA) ---
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    plt.scatter(components[idx, 0], components[idx, 1], label=f'Cluster {cluster}')
"""  ไม่เอาชื่อ
for i, name in enumerate(df['Name']):
    plt.text(components[i, 0], components[i, 1], name, fontsize=6)
"""  
plt.title('F1 Driver Clustering (Per Race)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
"""ver 3 : แก้feasure+แสดงข้อมูลในแต่ละcluster"""
"""
ความเร็วสูงสุดของนักแข่งในสนามนั้น
ความเร็วเฉลี่ยของนักแข่งในสนามนั้น
ตำแหน่งสตาร์ทของนักแข่งเฉลี่ยตั้งแต่2021ถึง2024
ตำแหน่งเข้าเส้นชัยของนักแข่งเฉลี่ยตั้งแต่2021ถึง2024
"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#เปิดระบบ cache
fastf1.Cache.enable_cache('cache')

data = []
names = []

# เก็บตำแหน่ง start/finish ของแต่ละนักแข่งรวมทุกสนาม
position_data = {}

# รอบแรก: รวบรวมข้อมูลทั้งหมดก่อน
for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year) #ดึงตารางการแข่งขันทั้งหมดของปีนั้น
    for _, row in schedule.iterrows(): #(ตัวแปร).iterrows() เป็นฟังก์ชันของ pandas.DataFrame ที่ใช้ วนลูปผ่านแต่ละแถว (row)
        if row['EventFormat'] != 'conventional':
            continue #ข้ามถ้าไม่ใช่conventional (conventional คือ รูปแบบการแข่งขันแบบปกติ ที่ไม่มี Sprint Race )
        try:
            session = fastf1.get_session(year, row['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue #ถ้านักแข่งคนนั้นไม่มีข้อมูลdata labก็ข้ามไป เนื่องจากต้องหาความเร็วสูงสุด

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue #ถ้าแข่งไม่จบก็ข้าม

            if drv not in position_data: #ถ้ายังไม่เคยเจอชื่อนี้ให้สร้างdictเปล่าไว้
                position_data[drv] = {
                    'Grid': [],
                    'Finish': []
                }
            #เตรียมไว้รอเอาไปเฉลี่ย
            position_data[drv]['Grid'].append(result['GridPosition'])
            position_data[drv]['Finish'].append(result['Position'])

# รอบสอง: ใช้ข้อมูลมาแบ่งกลุ่มแบบ per-race
for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year)
    for _, row in schedule.iterrows():
        if row['EventFormat'] != 'conventional':
            continue
        try:
            session = fastf1.get_session(year, row['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            #feasureเองงับเบ้บ
            max_speed = laps['SpeedST'].max()
            avg_speed = laps['SpeedST'].mean()
            # ดึงค่าเฉลี่ยตำแหน่งตลอดปีตั้งแต่2021ถึง2024
            avg_grid = np.mean(position_data[drv]['Grid'])
            avg_finish = np.mean(position_data[drv]['Finish'])

            #เกือบลืมมม names data สร้างไว้แต่แรกละ รอเก็บข้อมูลfeasureของเรา
            names.append(f"{year} {row['EventName']} - {session.get_driver(drv)['FullName']}") #เช่น 2023 Monaco GP - Lewis Hamilton ท่านเซอร์สุดคิ้วท์
            data.append([max_speed, avg_speed, avg_grid, avg_finish])

# สร้าง DataFrame
df = pd.DataFrame(data, columns=['MaxSpeed', 'AvgSpeed', 'AvgGridPos', 'AvgFinishPos'])
df['Name'] = names

# เอาข้อมูลจาก dataframe มาทำ Clustering 
X = df[['MaxSpeed', 'AvgSpeed', 'AvgGridPos', 'AvgFinishPos']]
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X) #จะได้ผลลัพธ์ว่าแถวนี้อยู่คลัสเตอร์ไหน (0–3)

# ตั้งชื่อคลัสเตอร์
df['ClusterLabel'] = df['Cluster'].map({
    0: 'lingBe',
    1: 'Beling',
    2: 'lingpenBe',
    3: 'Bepenling'
})

# PCA Visualization 
pca = PCA(n_components=2)
components = pca.fit_transform(X) #.fit() = ให้ PCA เรียนรู้จากข้อมูลว่า "ข้อมูลนี้กระจายไปในทิศทางไหนบ้าง" #.transform() = แปลงข้อมูลจาก 4 มิติ ➝ 2 มิติ (หรือเท่า n_components ที่กำหนดไว้)

plt.figure(figsize=(10, 6))

"""แบบนี้ไม่แสดงชื่อที่ตั้ง
for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    plt.scatter(components[idx, 0], components[idx, 1], label=f'Cluster {cluster}')
"""

for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    label = df[df['Cluster'] == cluster]['ClusterLabel'].iloc[0]  # ดึงชื่อคลัสเตอร์จริง
    plt.scatter(components[idx, 0], components[idx, 1], label=label)

# ถ้าไม่อยากโชว์ชื่อก็คอมเมนต์บรรทัดนี้ไว้
# for i, name in enumerate(df['Name']):
#     plt.text(components[i, 0], components[i, 1], name, fontsize=6)

plt.title(f'F1 Driver Clustering (2021–2024) - points : {len(df)}')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout() #เพื่อให้ไม่ให้ข้อความ (เช่น ชื่อแกน, title, label ฯลฯ) ซ้อนทับกันหรือโดนตัดออกจากกรอบรูป
plt.show()

#แสดงข้อมูลในแต่ละcluster
# --- แสดงข้อมูลนักแข่งในแต่ละ Cluster ---
"""อันนี้ไม่แสดงชื่อ
for cluster_id in sorted(df['Cluster'].unique()):
    print(f"\n🏁 Cluster {cluster_id} Driver Stats:\n")
    cluster_df = df[df['Cluster'] == cluster_id][
        ['Name', 'MaxSpeed', 'AvgSpeed', 'AvgGridPos', 'AvgFinishPos']
    ].sort_values(by='Name')

    print(cluster
    _df.to_string(index=False))
"""

# --- แสดงข้อมูลนักแข่งในแต่ละ Cluster พร้อมชื่อคลัสเตอร์ที่ตั้งเอง ---
for cluster_id in sorted(df['Cluster'].unique()): #.unique() เป็นเมธอดของ Pandas Series (หรือคอลัมน์ใน DataFrame) ที่ใช้สำหรับดึงค่า ที่ไม่ซ้ำกัน (unique values) ออกมาในรูปแบบ array (numpy array)
    cluster_label = df[df['Cluster'] == cluster_id]['ClusterLabel'].iloc[0] #.iloc[0] หมายถึงเอาแถวแรกที่เจอมาแสดง (เพราะทุกแถวในคลัสเตอร์เดียวกันจะมี label เดียวกันอยู่แล้ว)
    print(f"\n🏁 Cluster {cluster_id} - {cluster_label} Driver Stats:\n")

    cluster_df = df[df['Cluster'] == cluster_id][
        ['Name', 'MaxSpeed', 'AvgSpeed', 'AvgGridPos', 'AvgFinishPos']
    ].sort_values(by='Name')

    print(cluster_df.to_string(index=False))
# %%
"""ver 4 : feasure ที่เบอยากได้ 'คนมี 1 จุดข้อมูลต่อปี' """
"""
ปี2021ถึง2024
ความเร็วสูงสุดของนักแข่งเฉลี่ยทุกสนามในปีนั้น ๆ 
ความเร็วของนักแข่งเฉลี่ยทุกสนามในปีนั้น ๆ 
ตำแหน่งสตาร์ทของนักแข่งเฉลี่ยในแต่ละปี
ตำแหน่งเข้าเส้นชัยของนักแข่งเฉลี่ยในแต่ละปี
"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

# เก็บข้อมูลในรูปแบบ per-driver per-year
yearly_driver_data = {}

for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year)

    for _, row in schedule.iterrows():
        if row['EventFormat'] != 'conventional':
            continue  # ข้าม Sprint

        try:
            session = fastf1.get_session(year, row['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue  # ข้ามสนามที่ฝนตก

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            max_speed = laps['SpeedST'].max()
            avg_speed = laps['SpeedST'].mean()
            grid_pos = result['GridPosition']
            finish_pos = result['Position']

            key = (year, drv)
            if key not in yearly_driver_data:
                yearly_driver_data[key] = {
                    'MaxSpeeds': [],
                    'AvgSpeeds': [],
                    'GridPositions': [],
                    'FinishPositions': [],
                    'Name': session.get_driver(drv)['FullName']
                }

            yearly_driver_data[key]['MaxSpeeds'].append(max_speed)
            yearly_driver_data[key]['AvgSpeeds'].append(avg_speed)
            yearly_driver_data[key]['GridPositions'].append(grid_pos)
            yearly_driver_data[key]['FinishPositions'].append(finish_pos)

# --- ทำ DataFrame ---
records = []
for (year, drv), values in yearly_driver_data.items():
    records.append({
        'Year': year,
        'Driver': values['Name'],
        'AvgMaxSpeed': np.mean(values['MaxSpeeds']),
        'AvgAvgSpeed': np.mean(values['AvgSpeeds']),
        'AvgGridPos': np.mean(values['GridPositions']),
        'AvgFinishPos': np.mean(values['FinishPositions'])
    })

df = pd.DataFrame(records)

# --- Clustering ---
features = ['AvgMaxSpeed', 'AvgAvgSpeed', 'AvgGridPos', 'AvgFinishPos']
X = df[features]
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- ตั้งชื่อคลัสเตอร์ตามใจชอบ ---
df['ClusterLabel'] = df['Cluster'].map({
    0: 'Doraemon',
    1: 'Nobita',
    2: 'Takeshi',
    3: 'Bakkembe'
})

# --- แสดงผลแบบกราฟ (PCA) ---
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))

for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    label = df[df['Cluster'] == cluster]['ClusterLabel'].iloc[0]
    plt.scatter(components[idx, 0], components[idx, 1], label=label)

""" plotละดำเกิร
for i, row in df.iterrows():
    plt.text(components[i, 0], components[i, 1], f"{row['Year']} - {row['Driver']}", fontsize=6)
"""

plt.title(f'F1 Driver Clustering (2021–2024) - points : {len(df)}')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- แสดงข้อมูลในแต่ละคลัสเตอร์ ---
for cluster_id in sorted(df['Cluster'].unique()):
    label = df[df['Cluster'] == cluster_id]['ClusterLabel'].iloc[0]
    print(f"\n🏁 Cluster {cluster_id} - {label}:\n")
    cluster_df = df[df['Cluster'] == cluster_id][
        ['Year', 'Driver', 'AvgMaxSpeed', 'AvgAvgSpeed', 'AvgGridPos', 'AvgFinishPos']
    ].sort_values(by=['Year', 'Driver'])
    print(cluster_df.to_string(index=False))


"""ver 5 : เวอร์สุดท้ายของแทร่ ของจริว แบบมาก ๆ มากกกกกก 
พิจารณาปี2021ถึง2024 สนามที่ฝนไม่ตกมีการแข่งขันจนจบ race
feasure
SD ของ position ในปีนั้น ๆ 1 จุด / คน / ปี
ค่าเฉลี่ยการเบรกตอนเข้าโค้ง
ความเร็วเฉลี่ยตลอดทั้งปี
RPM เฉลี่ยต่อปี
จำนวนการใช้DRS
"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

yearly_data = {}

for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year)
    pos_diffs = []

    for _, row in schedule.iterrows():
        if row['EventFormat'] != 'conventional':
            continue

        try:
            session = fastf1.get_session(year, row['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            drv_data = yearly_data.setdefault((year, drv), {
                'BrakeCount': 0,
                'TotalLaps': 0,
                'SpeedList': [],
                'RpmList': [],
                'DrsUsage': 0,
                'DrsPossible': 0,
                'PosDiffList': [],
                'Name': session.get_driver(drv)['FullName']
            })

            # เพิ่ม position diff (grid - finish)
            pos_diff = result['GridPosition'] - result['Position']
            drv_data['PosDiffList'].append(pos_diff)
            pos_diffs.append(pos_diff)

            for lap in laps.iterlaps():
                tel = lap[1].get_telemetry()
                braking = tel[(tel['Brake'] == True) & (tel['Throttle'] == 0)]
                drv_data['BrakeCount'] += len(braking)
                drv_data['SpeedList'].extend(tel['Speed'].dropna())
                drv_data['RpmList'].extend(tel['RPM'].dropna())
                drv_data['DrsUsage'] += tel['DRS'].fillna(0).gt(0).sum()
                drv_data['DrsPossible'] += tel['DRS'].notna().sum()
                drv_data['TotalLaps'] += 1

    # คำนวณค่าเฉลี่ยของ pos_diff ทั้งปี
    avg_year_pos_diff = np.mean(pos_diffs)
    for (yr, drv), data in yearly_data.items():
        if yr == year:
            data['PosDiffSTD'] = np.std(np.array(data['PosDiffList']) - avg_year_pos_diff)

# --- สร้าง DataFrame ---
records = []
for (year, drv), values in yearly_data.items():
    drs_pct = (values['DrsUsage'] / values['DrsPossible']) * 100 if values['DrsPossible'] > 0 else 0
    records.append({
        'Year': year,
        'Driver': values['Name'],
        'PosDiffSTD': values['PosDiffSTD'],
        'BrakePerCorner': values['BrakeCount'] / values['TotalLaps'] if values['TotalLaps'] else 0,
        'AvgSpeed': np.mean(values['SpeedList']),
        'AvgRPM': np.mean(values['RpmList']),
        'DRSUsagePct': drs_pct
    })

df = pd.DataFrame(records)

# --- Clustering ---
features = ['PosDiffSTD', 'BrakePerCorner', 'AvgSpeed', 'AvgRPM', 'DRSUsagePct']
X = df[features]
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# ตั้งชื่อกลุ่ม
df['ClusterLabel'] = df['Cluster'].map({
    0: 'Tactical',
    1: 'Aggressor',
    2: 'Strategist',
    3: 'Speedster'
})

# --- PCA Visualization ---
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    label = df[df['Cluster'] == cluster]['ClusterLabel'].iloc[0]
    plt.scatter(components[idx, 0], components[idx, 1], label=label)

plt.title('F1 Driver Clustering (Per Year, 2021–2024)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- แสดงข้อมูลแต่ละคลัสเตอร์ ---
for cluster_id in sorted(df['Cluster'].unique()):
    label = df[df['Cluster'] == cluster_id]['ClusterLabel'].iloc[0]
    print(f"\n🏁 Cluster {cluster_id} - {label}:\n")
    print(df[df['Cluster'] == cluster_id].sort_values(by=['Year', 'Driver']).to_string(index=False))

# %%
"""ver 6 : ver 5 ตัด rpm ทิ้ง"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

driver_stats = {}

for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():
        if event['EventFormat'] != 'conventional':
            continue

        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        avg_diff = []
        all_driver_diffs = {}

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            start_pos = result['GridPosition']
            finish_pos = result['Position']
            pos_diff = start_pos - finish_pos

            avg_diff.append(pos_diff)
            all_driver_diffs[drv] = pos_diff

        if not avg_diff:
            continue

        mean_diff = np.mean(avg_diff)

        for drv, pos_diff in all_driver_diffs.items():
            laps = session.laps.pick_driver(drv)
            laps = laps.pick_quicklaps()
            if laps.empty:
                continue

            corners = 0
            brake_events = 0
            drs_usage = 0
            drs_total = 0
            avg_speeds = []

            for _, lap in laps.iterrows():
                tel = lap.get_telemetry()
                if tel.empty:
                    continue

                speed = tel['Speed'].to_numpy()
                brake = tel['Brake'].to_numpy()
                drs = tel['DRS'].to_numpy()

                avg_speeds.append(np.mean(speed))

                for i in range(1, len(speed)):
                    if speed[i] < speed[i-1] and brake[i]:
                        brake_events += 1
                        corners += 1

                drs_usage += np.sum(drs == 5)
                drs_total += len(drs)

            if drv not in driver_stats:
                driver_stats[drv] = {'Yearly': {}}

            if year not in driver_stats[drv]['Yearly']:
                driver_stats[drv]['Yearly'][year] = {
                    'PosDiffs': [],
                    'BrakeCount': [],
                    'AvgSpeed': [],
                    'DRSUsage': []
                }

            driver_stats[drv]['Yearly'][year]['PosDiffs'].append(pos_diff - mean_diff)
            driver_stats[drv]['Yearly'][year]['BrakeCount'].append(brake_events)
            driver_stats[drv]['Yearly'][year]['AvgSpeed'].extend(avg_speeds)
            if drs_total > 0:
                driver_stats[drv]['Yearly'][year]['DRSUsage'].append(drs_usage / drs_total * 100)

# สร้าง DataFrame
records = []
for drv, data in driver_stats.items():
    for year, values in data['Yearly'].items():
        records.append({
            'Driver': drv,
            'Year': year,
            'SD_PosDiff': np.std(values['PosDiffs']),
            'AvgBrakeCount': np.mean(values['BrakeCount']),
            'AvgSpeed': np.mean(values['AvgSpeed']),
            'AvgDRSUsage': np.mean(values['DRSUsage'])
        })

df = pd.DataFrame(records)
df.dropna(inplace=True)

# Clustering
features = ['SD_PosDiff', 'AvgBrakeCount', 'AvgSpeed', 'AvgDRSUsage']
X = df[features]

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# PCA Visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    plt.scatter(components[idx, 0], components[idx, 1], label=f'Cluster {cluster}')

plt.title('F1 Driver Clustering (2021–2024)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""ver 7 : ver 5 มี Elbow Method มีrpm"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

yearly_data = {}

for year in range(2023, 2025):
    schedule = fastf1.get_event_schedule(year)
    pos_diffs = []

    for _, row in schedule.iterrows():
        if row['EventFormat'] != 'conventional':
            continue

        try:
            session = fastf1.get_session(year, row['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            drv_data = yearly_data.setdefault((year, drv), {
                'BrakeCount': 0,
                'TotalLaps': 0,
                'SpeedList': [],
                'RpmList': [],
                'DrsUsage': 0,
                'DrsPossible': 0,
                'PosDiffList': [],
                'Name': session.get_driver(drv)['FullName']
            })

            # เพิ่ม position diff (grid - finish)
            pos_diff = result['GridPosition'] - result['Position']
            drv_data['PosDiffList'].append(pos_diff)
            pos_diffs.append(pos_diff)

            for lap in laps.iterlaps():
                tel = lap[1].get_telemetry()
                braking = tel[(tel['Brake'] == True) & (tel['Throttle'] == 0)]
                drv_data['BrakeCount'] += len(braking)
                drv_data['SpeedList'].extend(tel['Speed'].dropna())
                drv_data['RpmList'].extend(tel['RPM'].dropna())
                drv_data['DrsUsage'] += tel['DRS'].fillna(0).gt(0).sum()
                drv_data['DrsPossible'] += tel['DRS'].notna().sum()
                drv_data['TotalLaps'] += 1

    # คำนวณค่าเฉลี่ยของ pos_diff ทั้งปี
    avg_year_pos_diff = np.mean(pos_diffs)
    for (yr, drv), data in yearly_data.items():
        if yr == year:
            data['PosDiffSTD'] = np.std(np.array(data['PosDiffList']) - avg_year_pos_diff)

# --- สร้าง DataFrame ---
records = []
for (year, drv), values in yearly_data.items():
    drs_pct = (values['DrsUsage'] / values['DrsPossible']) * 100 if values['DrsPossible'] > 0 else 0
    records.append({
        'Year': year,
        'Driver': values['Name'],
        'PosDiffSTD': values['PosDiffSTD'],
        'BrakePerCorner': values['BrakeCount'] / values['TotalLaps'] if values['TotalLaps'] else 0,
        'AvgSpeed': np.mean(values['SpeedList']),
        'AvgRPM': np.mean(values['RpmList']),
        'DRSUsagePct': drs_pct
    })

df = pd.DataFrame(records)

# --- Clustering ---
features = ['PosDiffSTD', 'BrakePerCorner', 'AvgSpeed', 'AvgRPM', 'DRSUsagePct']
X = df[features]

# --- Elbow Method ---
inertia = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'o-', color='purple')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# ตั้งชื่อกลุ่ม
df['ClusterLabel'] = df['Cluster'].map({
    0: 'Tactical',
    1: 'Aggressor',
    2: 'Strategist',
    3: 'Speedster'
})

# --- PCA Visualization ---
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    label = df[df['Cluster'] == cluster]['ClusterLabel'].iloc[0]
    plt.scatter(components[idx, 0], components[idx, 1], label=label)

plt.title('F1 Driver Clustering (Per Year, 2021–2024)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- แสดงข้อมูลแต่ละคลัสเตอร์ ---
for cluster_id in sorted(df['Cluster'].unique()):
    label = df[df['Cluster'] == cluster_id]['ClusterLabel'].iloc[0]
    print(f"\n🏁 Cluster {cluster_id} - {label}:\n")
    print(df[df['Cluster'] == cluster_id].sort_values(by=['Year', 'Driver']).to_string(index=False))


#%%
"""ver 8 : ver 6 มี Elbow Method"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

driver_stats = {}

for year in range(2021, 2025):
    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():
        if event['EventFormat'] != 'conventional':
            continue

        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        avg_diff = []
        all_driver_diffs = {}

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            start_pos = result['GridPosition']
            finish_pos = result['Position']
            pos_diff = start_pos - finish_pos

            avg_diff.append(pos_diff)
            all_driver_diffs[drv] = pos_diff

        if not avg_diff:
            continue

        mean_diff = np.mean(avg_diff)

        for drv, pos_diff in all_driver_diffs.items():
            laps = session.laps.pick_driver(drv)
            laps = laps.pick_quicklaps()
            if laps.empty:
                continue

            corners = 0
            brake_events = 0
            drs_usage = 0
            drs_total = 0
            avg_speeds = []

            for _, lap in laps.iterrows():
                tel = lap.get_telemetry()
                if tel.empty:
                    continue

                speed = tel['Speed'].to_numpy()
                brake = tel['Brake'].to_numpy()
                drs = tel['DRS'].to_numpy()

                avg_speeds.append(np.mean(speed))

                for i in range(1, len(speed)):
                    if speed[i] < speed[i-1] and brake[i]:
                        brake_events += 1
                        corners += 1

                drs_usage += np.sum(drs == 5)
                drs_total += len(drs)

            if drv not in driver_stats:
                driver_stats[drv] = {'Yearly': {}}

            if year not in driver_stats[drv]['Yearly']:
                driver_stats[drv]['Yearly'][year] = {
                    'PosDiffs': [],
                    'BrakeCount': [],
                    'AvgSpeed': [],
                    'DRSUsage': []
                }

            driver_stats[drv]['Yearly'][year]['PosDiffs'].append(pos_diff - mean_diff)
            driver_stats[drv]['Yearly'][year]['BrakeCount'].append(brake_events)
            driver_stats[drv]['Yearly'][year]['AvgSpeed'].extend(avg_speeds)
            if drs_total > 0:
                driver_stats[drv]['Yearly'][year]['DRSUsage'].append(drs_usage / drs_total * 100)

# สร้าง DataFrame
records = []
for drv, data in driver_stats.items():
    for year, values in data['Yearly'].items():
        records.append({
            'Driver': drv,
            'Year': year,
            'SD_PosDiff': np.std(values['PosDiffs']),
            'AvgBrakeCount': np.mean(values['BrakeCount']),
            'AvgSpeed': np.mean(values['AvgSpeed']),
            'AvgDRSUsage': np.mean(values['DRSUsage'])
        })

df = pd.DataFrame(records)
df.dropna(inplace=True)

# Clustering
features = ['SD_PosDiff', 'AvgBrakeCount', 'AvgSpeed', 'AvgDRSUsage']
X = df[features]

#elbow
# Elbow Method เพื่อเลือกจำนวนคลัสเตอร์ที่เหมาะสม
inertias = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# PCA Visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for cluster in df['Cluster'].unique():
    idx = df['Cluster'] == cluster
    plt.scatter(components[idx, 0], components[idx, 1], label=f'Cluster {cluster}')

plt.title('F1 Driver Clustering (2021–2024)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()








"""ver 7.5 : ฉบับเซ้นเอาไปแก้errorตอนทำโมเดลให้แล้ว"
#แบ่งได้เป็นสามคลัสเตอร์ สิ่งที่กำลังจะแก้ต่อไป คือเพิ่มชื่อนักแข่ง 
#และก็เดี๋ยวดูตรงsd อยากให้มันมี+- เพื่อดูการแซงการโดนแซง //จากอิเบ
#ใช้สูตร (deltaxi - x_bar )/sd //จากหมีภู
#แต่ถ้าทำไม่ได้ ค่อยไปพิจารณาตรงอื่นควบฟีเจอร์ งิงิ //จากเซ้นส์
import fastf1
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('cache')

yearly_data = {}

for year in range(2023, 2025):
    schedule = fastf1.get_event_schedule(year)
    pos_diffs = []

    for _, row in schedule.iterrows():
        if row['EventFormat'] != 'conventional':
            continue

        try:
            session = fastf1.get_session(year, row['RoundNumber'], 'R')
            session.load()
        except:
            continue

        if session.weather_data['Rainfall'].sum() > 0:
            continue

        for drv in session.drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty or drv not in session.results.index:
                continue

            result = session.results.loc[drv]
            if result['Status'] != 'Finished':
                continue

            drv_data = yearly_data.setdefault((year, drv), {
                'BrakeCount': 0,
                'TotalLaps': 0,
                'SpeedList': [],
                'RpmList': [],
                'DrsUsage': 0,
                'DrsPossible': 0,
                'PosDiffList': [],
                'Name': session.get_driver(drv)['FullName']
            })

            # เพิ่ม position diff (grid - finish)
            pos_diff = result['GridPosition'] - result['Position']
            drv_data['PosDiffList'].append(pos_diff)
            pos_diffs.append(pos_diff)


        for lap in laps.iterlaps():
                tel = lap[1].get_telemetry()
                braking = tel[(tel['Brake'] == True) & (tel['Throttle'] == 0)]
                drv_data['BrakeCount'] += len(braking)
                drv_data['SpeedList'].extend(tel['Speed'].dropna())
                drv_data['RpmList'].extend(tel['RPM'].dropna())
                drv_data['DrsUsage'] += tel['DRS'].fillna(0).gt(0).sum()
                drv_data['DrsPossible'] += tel['DRS'].notna().sum()
                drv_data['TotalLaps'] += 1

    # คำนวณค่าเฉลี่ยของ pos_diff ทั้งปี
    avg_year_pos_diff = np.mean(pos_diffs)
    for (yr, drv), data in yearly_data.items():
        if yr == year:
            data['PosDiffSTD'] = np.std(np.array(data['PosDiffList']) - avg_year_pos_diff)

# --- สร้าง DataFrame ---
records = []
for (year, drv), values in yearly_data.items():
    drs_pct = (values['DrsUsage'] / values['DrsPossible']) * 100 if values['DrsPossible'] > 0 else 0
    records.append({
        'Year': year,
        'Driver': values['Name'],
        'PosDiffSTD': values['PosDiffSTD'],
        'BrakePerCorner': values['BrakeCount'] / values['TotalLaps'] if values['TotalLaps'] else 0,
        'AvgSpeed': np.mean(values['SpeedList']),
        'AvgRPM': np.mean(values['RpmList']),
        'DRSUsagePct': drs_pct
    })


df = pd.DataFrame(records)

#%%


# --- Clustering ---
features = ['PosDiffSTD', 'BrakePerCorner', 'AvgSpeed', 'AvgRPM', 'DRSUsagePct']
X = df[features]

# --- Elbow Method ---
inertia = []
K_range = range(1, 10)


# Remove rows with missing feature data once
X = df[features].dropna()
if X.empty:
    print("X is empty — no complete rows to cluster.")
else:
    inertia = []
    K_range = range(1, 10)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'o-', color='purple')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(K_range)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



kmeans = KMeans(n_clusters=4, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X)

# ตั้งชื่อกลุ่ม
df_clean['ClusterLabel'] = df_clean['Cluster'].map({
    0: 'Tactical',
    1: 'Aggressor',
    2: 'Strategist',
    3: 'Speedster'
})

pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for cluster in df_clean['Cluster'].unique():
    idx = df_clean['Cluster'] == cluster
    label = df_clean[df_clean['Cluster'] == cluster]['ClusterLabel'].iloc[0]
    plt.scatter(components[idx, 0], components[idx, 1], label=label)

plt.title('F1 Driver Clustering (Per Year, 2021–2024)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- แสดงข้อมูลแต่ละคลัสเตอร์ ---
for cluster_id in sorted(df_clean['Cluster'].unique()):
    label = df_clean[df_clean['Cluster'] == cluster_id]['ClusterLabel'].iloc[0]
    print(f"\n🏁 Cluster {cluster_id} - {label}:\n")
    print(df_clean[df_clean['Cluster'] == cluster_id].sort_values(by=['Year', 'Driver']).to_string(index=False))
