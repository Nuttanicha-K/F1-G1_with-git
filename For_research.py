import fastf1
# Load a race session (for example, 2023 Australian Grand Prix)
session = fastf1.get_session(2023, 'Australian', 'R')

# Load the session data (including incidents, laps, etc.)
session.load()

# Get incident data (if available)
incidents = session.incidents

# Print all incidents
for incident in incidents:
    print(incident)
