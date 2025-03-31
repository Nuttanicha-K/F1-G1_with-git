# %%
import fastf1


# %%
#ลองดึงข้อมูลโบกธงดูเฉย ๆ สงสัย
session = fastf1.get_session(2023, 'Australian', 'R')
session.load()
track_status = session.track_status
print(track_status)


