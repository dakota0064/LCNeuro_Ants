import numpy as np

class TrophallaxisEvent():

    def __init__(self, ants, event_list, initial_time, final_time):
        # Event list will be a list of the form (timestamp, center_point)
        # Holds the individual frames comprising this interaction
        self.event_list = event_list
        # Initial time is a double marking the first frame of the event
        self.initial_time = initial_time
        # Final time is a double marking the last frame of the event
        self.final_time = final_time

        # A list of the ants (integers) involved in this event
        self.ants = ants


    # Compares the new record with all existing records
    # If overlap is found, adds this records' information in with the located event
    def compare_combine(self, event):
        # Check if these events share all the same ants
        for ant in self.ants:
            if ant not in event.ants:
                return False

        # Check if the event times overlap - if so, combine them
        # -8 ensures that we still combine frames within 3 timesteps of each other (2 frames per), to combat flickering from tagging
        if self.final_time - event.initial_time > -20 and event.final_time - self.initial_time > -20:

            # Reset final time and initial time accordingly
            event.final_time = max(self.final_time, event.final_time)
            event.initial_time = min(self.initial_time, event.initial_time)

            # Combine ants where applicable
            for ant in self.ants:
                if ant not in event.ants:
                    event.ants.append(ant)

            # Combine event lists where applicable (ignores ghost frames that get combined over)
            # Assumes events are sorted by timestamp
            if self.event_list[0][0] < event.event_list[0][0]:
                starting_point = event.event_list[0][0]
                cutoff = self.event_list[-1][0]
            else:
                starting_point = self.event_list[0][0]
                cutoff = event.event_list[-1][0]

            event_list = self.event_list + event.event_list
            if len(event_list) > 1:
                event_list.sort(key=lambda x: x[0])

            combined_events = []
            for i in range(len(event_list)):
                points = []
                timestep = event_list[i][0]
                if timestep < starting_point or timestep > cutoff:
                    combined_events.append(event_list[i])
                    continue

                else:
                    # Only do this for the area in between the cutoffs
                    points.append(event_list[i][1])
                    for j in range(i+1, len(event_list)):
                        if event_list[j][0] == timestep:
                            i = j
                            points.append(event_list[i][1])
                        elif event_list[j][0] > timestep:
                            break

                    center_point = np.mean(points, axis=0)
                    combined_events.append((timestep, center_point))
                    break

            event.event_list = combined_events

            return True



