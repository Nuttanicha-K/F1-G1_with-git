#%%
import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
import pandas as pd
import fastf1.plotting
import numpy as np
import sklearn

#%%
fastf1.Cache.enable_cache('cache')

years = range(2021, 2026)

rainy_drivers = set()
rainy_events = []

for year in years:
    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():
        try:
            #พิจารณาแค่raceพอ
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load()

            #เช็กสภาพอากาศครัฟ
            weather = session.weather_data
            if weather is not None and (weather['Rainfall'] > 0).any():
                rainy_events.append((year, event['EventName']))

                # เอาชื่อนักแข่งฮ๊่ฟ
                results = session.results
                if results is not None:
                    for _, row in results.iterrows():
                        full_name = row.get('FullName')
                        if full_name:
                            rainy_drivers.add(full_name)

        except Exception as e:
            print(f"Skipped {year} {event['EventName']} (R): {e}")


print("\n🌧️ Rainy Races from 2021–2025:")
for yr, ev in rainy_events:
    print(f"{yr} - {ev}")

print("\n👨‍🏁 Drivers who raced in rainy races:")
for name in sorted(rainy_drivers):
    print(name)

 
#%%
#เอาเวลาที่ธงเหลืองกับรถฉุกเฉินวิ่งในแมตช์ที่มีฝนตกตั้งแต่ปี 2021-2025
import fastf1
from datetime import datetime

fastf1.Cache.enable_cache('cache')
years = range(2021, 2023)
rainy_yellow_flag_data = []

today = datetime.today()

for year in years:
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventDate'] < today] 

    for _, event in schedule.iterrows():
        try:
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load()

            # Check if it was a rainy race
            weather = session.weather_data
            if weather is not None and (weather['Rainfall'] > 0).any():
                
                track_status = session.track_status

                if track_status is not None:
                    yellow_safety = track_status[track_status['Status'].isin(['2', '4'])]

                    # Only include if there were such events
                    if not yellow_safety.empty:
                        rainy_yellow_flag_data.append({
                            'Year': year,
                            'Event': event['EventName'],
                            'Yellow/Safety Times': list(yellow_safety['Time'])
                        })

        except Exception as e:
            print(f"Skipped {year} {event['EventName']} (R): {e}")

print("\n🟡 Rainy Races with Yellow Flag or Safety Car:")
for item in rainy_yellow_flag_data:
    print(f"{item['Year']} - {item['Event']}")
    for t in item['Yellow/Safety Times']:
        print(f"   ⏱️ {t}")

# %%
import fastf1
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#ทำการแบ่งคลัสเตอร์กันฮ๊าฟฟฟฟฟฟฟ
fastf1.Cache.enable_cache('cache')
years = range(2021, 2023) #คอมเส้นไหวแค่นี้ ทำต่อหน่อย2021-2026
today = datetime.today()

driver_data = []

for year in years:
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventDate'] < today]

    for _, event in schedule.iterrows():
        try: #เซ้นส์จะเอาแค่race
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load()
            #เอาแมตช์ที่ฝนตก
            weather = session.weather_data
            if weather is None or not (weather['Rainfall'] > 0).any():
                continue

            laps = session.laps
            results = session.results
            if laps is None or results is None:
                continue

            for _, row in results.iterrows():
                drv = row['Abbreviation']
                full_name = row['FullName']
                driver_laps = laps.pick_driver(drv)
                lap_count = len(driver_laps)
                total_time = driver_laps['LapTime'].sum() if not driver_laps.empty else pd.Timedelta(0)

                driver_data.append({
                    'Year': year,
                    'Event': event['EventName'],
                    'Driver': full_name,
                    'LapCount': lap_count,
                    'TotalTime': total_time.total_seconds(),
                    'GridPosition': row.get('GridPosition', None),
                    'RacePosition': row.get('Position', None),
                    'Status': row.get('Status'),
                    'DNF': 0 if row.get('Status') == 'Finished' else 1
                })

        except Exception as e:
            print(f"Skipped {year} {event['EventName']}: {e}")


df = pd.DataFrame(driver_data)
df.dropna(subset=['LapCount', 'TotalTime', 'GridPosition', 'RacePosition'], inplace=True)

#ฟีเจอร์ที่จะใช้ในคลัสเตอร์
features = df[['LapCount', 'TotalTime', 'GridPosition', 'RacePosition']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

#ใส่k-meanลงไปในคลัสเตอร์เราฮ๊าฟ
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Compare clusters to DNF label
print("\n🔍 Cluster Analysis vs. Actual DNF:")
print(df.groupby(['Cluster', 'DNF']).size())


plt.figure(figsize=(10, 6))
plt.scatter(df['LapCount'], df['TotalTime'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Lap Count")
plt.ylabel("Total Time (s)")
plt.title("Driver Clusters in Rainy Races (K-Means)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# %%
#ดูชื่อนักแข่งแบบโดยย่อ
for cluster_id in sorted(df['Cluster'].unique()):
    print(f"\nCluster {cluster_id}:")
    display(df[df['Cluster'] == cluster_id][['Driver', 'Event', 'LapCount', 'TotalTime', 'DNF']].head())

# %%
# โชว์ชื่อนักแข่งทั้งหมดในคลัสเตอร์นั้น ๆ
# เช็กว่าclusterไหนมีนักแข่งแข่งไม่จบเยอะกว่า จะได้ใส่ชื่อ
cluster_dnf_counts = df.groupby('Cluster')['DNF'].mean()

# เอาคลันเตอร์ที่มีเรตนักแข่งแข่งไม่จบมากกว่ามา
dnf_cluster = cluster_dnf_counts.idxmax()
finish_cluster = cluster_dnf_counts.idxmin()

#ใส่ชื่อคลัสเตอร์
df['ClusterLabel'] = df['Cluster'].map({
    dnf_cluster: 'Have high risk to not finished', #กลุ่มที่มีปัจจัยที่มีความเสี่ยงสูงว่าจะแข่งไม่จบ แต่ไม่ได้แปลว่าจะแข่งไม่จบ
    finish_cluster: 'Have low risk to not finished' #กลุ่มที่มีความเสี่ยงต่ำ แต่ไม่ได้แปลว่าจะแข่งจบ
})

# โชว์ชื่อนักแข่งทั้งหมด บริ๊น ๆๆๆๆๆ
for label in df['ClusterLabel'].unique():
    print(f"\n🚥 {label} Drivers:")
    drivers = df[df['ClusterLabel'] == label][['Driver', 'Event', 'Year', 'DNF']]
    print(drivers.sort_values(by=['Year', 'Event']).to_string(index=False))

#%%

for label in df['ClusterLabel'].unique():
    print(f"\n🚥 {label} Drivers:")
    drivers = df[df['ClusterLabel'] == label][['Driver', 'Event', 'Year', 'DNF', 'LapCount', 'TotalTime', 'GridPosition', 'RacePosition']]
    print(drivers.sort_values(by=['Year', 'Event']).to_string(index=False))