# %%
import fastf1


# %%
#ลองดึงข้อมูลโบกธงดูเฉย ๆ สงสัย
session = fastf1.get_session(2023, 'Australian', 'R')
session.load()
track_status = session.track_status
print(track_status)

#%%
"""เริ่มต้น: ติดตั้งและตั้งค่า"""
"""MP"""
# ติดตั้ง fastf1 ก่อนถ้ายังไม่มี
# pip install fastf1

import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt

# เปิดแคช # ต้องสร้างโฟลเดอร์แคช
fastf1.Cache.enable_cache('./cache')  

# โหลด session (เช่น รอบคัดเลือกของ Bahrain 2023)
session = fastf1.get_session(2023, 'Azerbaijan', 'Q')  # 'Q' = Qualifying, 'R' = Race
session.load()

# โหลด session 
session = fastf1.get_session(2023, 'Bahrain', 'Q')  # 'Q' = Qualifying, 'R' = Race
session.load()

# โหลด session 
session = fastf1.get_session(2024, 'Bahrain', 'R')  # 'Q' = Qualifying, 'R' = Race
session.load()

#%%
"""แสดงรายชื่อนักแข่งทั้งหมด"""
import fastf1
fastf1.Cache.enable_cache('./cache')  # อย่าลืมสร้าง cache ด้วย

# โหลด session
session = fastf1.get_session(2023, 'Bahrain', 'Q')  # 'Q' = Qualifying
session.load()  # ดึงข้อมูลมาใช้งาน

#เริ่ม **ใช้drivers = session.drivers ไม่ได้เนื่องจากบาง session (โดยเฉพาะ Qualifying/Practice) อาจ ไม่มีคีย์ 'Team' อยู่ในนั้น
results = session.results

print("นักแข่งใน session นี้:")
for _, row in results.iterrows():
    print(f"{row['Abbreviation']}: {row['FullName']} - {row['TeamName']}")

# %%
"""อยากรู้ปีนั้น มีสนามไหนบ้าง"""
from fastf1 import events

# ดูรายการสนามทั้งหมดของปี 2023
schedule = events.get_event_schedule(2023)

# แสดงรายการแบบอ่านง่าย
for index, row in schedule.iterrows():
    print(f"{row['EventName']} - {row['Country']} ({row['Location']}) | วันที่: {row['EventDate'].date()}")

# %%
"""แสดงเวลาต่อรอบของนักแข่งทั้งหมด"""
#โหลดsession ก่อน
import fastf1
import matplotlib.pyplot as plt

# เปิดใช้งาน cache
fastf1.Cache.enable_cache('./cache')

# โหลด session (ปี 2023 สนาม Bahrain รอบ Qualifying)
session = fastf1.get_session(2023, 'Bahrain', 'Q')
session.load()  # ต้องโหลดก่อนถึงจะเข้าถึงข้อมูลได้

#เริ่ม
laps = session.laps
# หารอบเร็วที่สุดของแต่ละนักแข่ง
fastest_laps = laps.pick_quicklaps().groupby('Driver').min().sort_values(by='LapTime')

print("เวลาต่อรอบที่เร็วที่สุดของแต่ละนักแข่ง:")
print(fastest_laps[['LapTime', 'Team']])

#%%
"""กราฟแสดงเวลาต่อรอบที่ดีที่สุด"""
plotting.setup_mpl()  # ตั้งค่าการแสดงผลให้เหมาะกับ F1

fastest_laps = fastest_laps.reset_index()

plt.figure(figsize=(10, 5))
plt.barh(fastest_laps['Driver'], fastest_laps['LapTime'].dt.total_seconds(), color='skyblue')
plt.xlabel("Lap Time (s)")
plt.title("Fastest Lap Time per Driver")
plt.gca().invert_yaxis()
plt.show()

#%%
"""ดูรายละเอียดเฉพาะของนักแข่งบางคน"""
# เช่น ดูข้อมูลรอบของ Max Verstappen
laps_ver = laps.pick_driver('VER')
print(laps_ver[['LapNumber', 'LapTime', 'Compound', 'Stint']])

# %%
"""วาดกราฟแสดง ความเร็วรถ (Speed) ของ Charles Leclerc ในรอบที่เร็วที่สุดของรอบ Qualifying ที่ Monza ปี 2019"""
from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting
#ตั้งค่าธีมของกราฟให้ดูแนว F1 เท่ ๆ (สีเหมือนกราฟจริง ๆ ของ F1TV)
fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
#session.load() = ดึงข้อมูลทั้งหมดของ session นั้น (เช่น lap, car data, telemetry)
session = fastf1.get_session(2019, 'Monza', 'Q')
session.load()

"""!!!ถ้าอยากเปรียบเทียบกับนักแข่งคนอื่น"""
# Leclerc
fast_leclerc = session.laps.pick_drivers('LEC').pick_fastest()
lec_car_data = fast_leclerc.get_car_data()
t = lec_car_data['Time']
vCar = lec_car_data['Speed']

# Hamilton
fast_ham = session.laps.pick_drivers('HAM').pick_fastest()
ham_car_data = fast_ham.get_car_data()
t2 = ham_car_data['Time']
v2 = ham_car_data['Speed']

# Plot ทั้งสองคนในกราฟเดียวกัน
fig, ax = plt.subplots()
ax.plot(t, vCar, label='Leclerc', color='red')
ax.plot(t2, v2, label='Hamilton', color='blue')

ax.set_xlabel('Time')
ax.set_ylabel('Speed [Km/h]')
ax.set_title('Fastest Lap Speed - Leclerc vs Hamilton')
ax.legend()
plt.show()

# %%
"""หาตำแหน่งรถ P"""
import fastf1
session = fastf1.get_session(2023, 'Monza', 'Q')  # หรือปี/สนามที่คุณต้องการ
session.load()

results = session.results
print(results[['Position','FullName']])

# %%
"""ปีนั้น เอาทุกสนามลงในcache"""
import fastf1
from time import sleep

# ✅ เปิด cache: ต้องใส่ก่อน load ทุกครั้ง
fastf1.Cache.enable_cache('cache')  # ชื่อโฟลเดอร์จะสร้างอัตโนมัติถ้ายังไม่มี

year = 2023
schedule = fastf1.get_event_schedule(year)

# 🏁 วนโหลดรอบ Race ('R') ของทุกสนามในปีนั้น
for i, row in schedule.iterrows():
    round_number = row['RoundNumber']
    event_name = row['EventName']
    
    try:
        print(f"\n📦 Loading RACE for {event_name} (Round {round_number})...")
        session = fastf1.get_session(year, round_number, 'R')
        session.load()  # 💾 โหลดและเซฟเข้า cache

        print("✅ Cached successfully.")
        sleep(2)  # ป้องกันโหลดถี่เกินจนโดน block

    except Exception as e:
        print(f"❌ Failed to load {event_name}: {e}")

# %%
"""check ว่าแต่ละ object ให้ข้อมูลอะไรได้บ้าง"""
import fastf1
session = fastf1.get_session(2023, 'Monza', 'Q')  # หรือปี/สนามที่คุณต้องการ
session.load()

#ตัวอย่าง object laps
session.laps.columns
#ตัวอย่าง object results
session.results.columns

#%%
import fastf1
from time import sleep

# ✅ เปิด cache: ต้องใส่ก่อน load ทุกครั้ง
fastf1.Cache.enable_cache('cache')  # ชื่อโฟลเดอร์จะสร้างอัตโนมัติถ้ายังไม่มี

year = 2021
schedule = fastf1.get_event_schedule(year)

# 🏁 วนโหลดรอบ Race ('R') ของทุกสนามในปีนั้น
for i, row in schedule.iterrows():
    round_number = row['RoundNumber']
    event_name = row['EventName']
    
    try:
        print(f"\n📦 Loading RACE for {event_name} (Round {round_number})...")
        session = fastf1.get_session(year, round_number, 'R')
        session.load()  # 💾 โหลดและเซฟเข้า cache

        print("✅ Cached successfully.")
        sleep(2)  # ป้องกันโหลดถี่เกินจนโดน block

    except Exception as e:
        print(f"❌ Failed to load {event_name}: {e}")

# %%
import fastf1
from time import sleep


fastf1.Cache.enable_cache('cache')

year = 2022
schedule = fastf1.get_event_schedule(year)

for i, row in schedule.iterrows():
    round_number = row['RoundNumber']
    event_name = row['EventName']
    
    try:
        print(f"\n📦 Loading RACE for {event_name} (Round {round_number})...")
        session = fastf1.get_session(year, round_number, 'R')
        session.load()  # 💾 โหลดและเซฟเข้า cache

        print("✅ Cached successfully.")
        sleep(2)  # ป้องกันโหลดถี่เกินจนโดน block

    except Exception as e:
        print(f"❌ Failed to load {event_name}: {e}")

#%%
import fastf1
from time import sleep

fastf1.Cache.enable_cache('cache')
year = 2024
schedule = fastf1.get_event_schedule(year)

for i, row in schedule.iterrows():
    round_number = row['RoundNumber']
    event_name = row['EventName']
    
    try:
        print(f"\n📦 Loading RACE for {event_name} (Round {round_number})...")
        session = fastf1.get_session(year, round_number, 'R')
        session.load()  # 💾 โหลดและเซฟเข้า cache

        print("✅ Cached successfully.")
        sleep(2)  # ป้องกันโหลดถี่เกินจนโดน block

    except Exception as e:
        print(f"❌ Failed to load {event_name}: {e}")

#%%
import fastf1
from time import sleep

fastf1.Cache.enable_cache('cache')
year = 2024
schedule = fastf1.get_event_schedule(year)
year = 2025
schedule = fastf1.get_event_schedule(year)

for i, row in schedule.iterrows():
    round_number = row['RoundNumber']
    event_name = row['EventName']
    
    try:
        print(f"\n📦 Loading RACE for {event_name} (Round {round_number})...")
        session = fastf1.get_session(year, round_number, 'R')
        session.load()  # 💾 โหลดและเซฟเข้า cache

        print("✅ Cached successfully.")
        sleep(2)  # ป้องกันโหลดถี่เกินจนโดน block

    except Exception as e:
        print(f"❌ Failed to load {event_name}: {e}")
