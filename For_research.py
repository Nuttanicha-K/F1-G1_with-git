import fastf1

# Load a session (example: 2023 Australian Grand Prix, race)
session = fastf1.get_session(2023, 'Australian', 'R')

# Load all the session data (including track status)
session.load()

# Access track status information
track_status = session.track_status

# Print the track status to see the details
print(track_status)

for status in track_status:
    print(f"Time: {status['time']}, Status: {status['status']}")
