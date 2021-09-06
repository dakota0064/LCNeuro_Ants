import matplotlib
#matplotlib.use("TKAGG")
#import matplotlib.pyplot as plt
from trophallaxis_list import TrophallaxisEvent
import timeit
import pandas as pd
import cv2
import numpy as np

########################################################################################################################
""" Take in a video file and csv file, and return a list of candidate trophallaxis events
    :returns an edited video"""

def find_trophallaxis_candidates(video_file, csv_file, trophallaxis_df, queen_number, foragers_A):
    candidates = []

    # Hyperparameters
    search_radius = 30

    # Frames are 1790 x 1200
    test_video = cv2.VideoCapture(video_file)
    good_read, frame = test_video.read()
    if not good_read:
        print("Error getting frame size from ", video_file)

    # video = cv2.VideoCapture(video_file)
    # writer = cv2.VideoWriter(video_file[:-4] + "_annotated.avi",
    #                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
    #                          (frame.shape[1], frame.shape[0]))

    df = pd.read_csv(csv_file)

    count = 0
    max_index = len(df["time[s] #"])

    column_headers = list(df.columns.values)
    ant_indices = set()

    # Get the indices of the individual ants, since they will each have a column header with a '-x' value
    for header in column_headers:
        if ("-x") in header:
            ant_indices.add(header[:-2])

    ant_indices = list(ant_indices)
    ant_indices.sort()

    while count < max_index:

        ant_heads = {}
        # # Get the current frame to mark up with boxes
        # good_read, frame = video.read()
        # if not good_read:
        #     print("Error reading frame ", count, " from video file ", video_file)
        # #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        head_points = []

        row = df.iloc[count]
        frame_idx = row["time[s] #"]

        for ant in ant_indices:
            # Initialize head center as (-1, -1)
            ant_heads[int(ant)] = (-1, -1)
            x = row[ant + "-x"]
            y = row[ant + "-y"]
            angle = row[ant + "-angle"]

            #print(x, y, angle)

            # If any -1 values, we have incomplete data for this ant at this frame
            if x == -1 or y == -1 or angle == -1:
                continue

            # Ants appear approx 100px long and 30px wide
            #Todo find a better way to compute this

            # Find the head by converting to polar coordinates
            head_x = int(x + (35 * np.cos(np.radians(angle))))
            head_y = int(y + (35 * np.sin(np.radians(angle))))

            if int(ant) == queen_number:
                head_x = int(x + (160 * np.cos(np.radians(angle+540))))
                head_y = int(y + (160 * np.sin(np.radians(angle+540))))
            #head_rect = ((head_x, head_y), (40, 40), angle)
            #head_box = cv2.boxPoints(head_rect)
            #head_box = np.int0(head_box)

            # rect = ((center_pt), (width, height), angle)
            rect = ((x, y), (100, 30), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw the box and head circle over the corresponding ant
            #cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
            #cv2.circle(frame, (head_x, head_y), search_radius, (0, 0, 255), 2)
            #cv2.drawContours(frame, [head_box], 0, (255, 0, 0), 2)

            # Save the head circle points for comparison to see if ants are kissing
            head_points.append((int(ant), np.array([head_x, head_y])))
            ant_heads[int(ant)] = (head_x, head_y)

        # Compare all points, redraw circle for candidate trophallaxis events
        for point in head_points:
            for other_point in head_points:
                if point[1][0] == other_point[1][0] and point[1][1] == other_point[1][1]:
                    continue
                else:
                    distance = np.linalg.norm(point[1]-other_point[1])
                    if distance < (search_radius * 2) - 1:

                        prime_ant = -1
                        secondary_ant = -1

                        ant1 = point[0]
                        ant2 = other_point[0]

                        if ant1 in foragers_A:
                            prime_ant = ant1
                            secondary_ant = ant2

                        if ant2 in foragers_A:
                            prime_ant = ant2
                            secondary_ant = ant1

                        # If neither are foragers, organize by crop size
                        if prime_ant == -1:
                            crop1 = int(row[str(ant1) + "-crop_intensity"])
                            crop2 = int(row[str(ant2) + "-crop_intensity"])

                            if crop1 > crop2:
                                prime_ant = ant1
                                secondary_ant = ant2

                            if crop2 > crop1:
                                prime_ant = ant1
                                secondary_ant = ant2

                        if prime_ant == -1:
                            prime_ant = ant1
                            secondary_ant = ant2

                        # Get components for a trophallaxis event
                        ants = [prime_ant, secondary_ant]
                        event_list = [[frame_idx, np.mean([point[1], other_point[1]], axis=1)]]
                        initial_time = frame_idx
                        final_time = frame_idx
                        event = TrophallaxisEvent(ants, event_list, initial_time, final_time)

                        combined = False
                        for i in range(len(candidates)):
                            combined = event.compare_combine(candidates[i])
                            if combined:
                                break

                        if not combined:
                            candidates.append(event)

                        # Draw green circle
                        #cv2.circle(frame, (point[1][0], point[1][1]), search_radius, (0, 255, 0), 2)

        # # Check if there were any trophallaxis events at this frame
        # events = trophallaxis_df.loc[trophallaxis_df["final_time"] >= frame_idx]
        # events = events.loc[events["initial_time"] <= frame_idx]
        # for index, row in events.iterrows():
        #     contour_list = []
        #     seed_ant = int(row["ant1"])
        #     if seed_ant == -1:
        #         seed_ant = int(row["ant2"])
        #         if seed_ant == -1:
        #             continue
        #     seed_ant_center = ant_heads[seed_ant]
        #     if seed_ant_center[0] == -1 or seed_ant_center[1] == -1:
        #         seed_ant = int(row["ant2"])
        #         if seed_ant > 0:
        #             ant_center = ant_heads[seed_ant]
        #             if seed_ant_center[0] == -1 or seed_ant_center[1] == -1:
        #                 continue
        #         else:
        #             continue
        #
        #     contour_list.append(seed_ant_center)
        #     for point in head_points:
        #         if seed_ant_center[0] == point[0] and seed_ant_center[1] == point[1]:
        #             continue
        #         else:
        #             distance = np.linalg.norm(seed_ant_center-point)
        #             if distance < (search_radius * 2) - 1:
        #                 contour_list.append(point + np.array([25, 25]))
        #                 contour_list.append(point + np.array([-25, 25]))
        #                 contour_list.append(point + np.array([25, -25]))
        #                 contour_list.append(point + np.array([-25, -25]))
        #
        #     contour_list = np.array(contour_list, dtype=np.int32)
        #
        #     # Sometimes the forager ant listed isn't actually tracked by the qr scanner
        #     # In that case we assume a bounding box around the foragers head
        #     if len(contour_list) == 1:
        #         point = contour_list[0]
        #         bounding_rect = [point[0]-25, point[1]-25, 50, 50]
        #     else:
        #         bounding_rect = cv2.boundingRect(contour_list)
        #
        #     cv2.rectangle(frame, (int(bounding_rect[0]), int(bounding_rect[1])),
        #                   (int(bounding_rect[0])+int(bounding_rect[2]),  int(bounding_rect[1]) + int(bounding_rect[3])),
        #                   (0, 255, 255), 5)

            # plt.imshow(frame)
            # plt.savefig("colony_A_bb.jpg")
            # plt.show()

        # writer.write(frame)


        #print(count)
        if (count+1) % 1000 == 0:
            print("Processed " + str(count+1) + " frames")
        count += 1

    #writer.release()
    return candidates

########################################################################################################################

if __name__ == '__main__':
    start_time = timeit.default_timer()

    colony_A_events = pd.read_csv("data/interactions_data_A.csv")
    forager_df = pd.read_csv("data/interactions_data_foragers.csv")
    foragers_A = list(set(forager_df[forager_df["Experiment"] == "A"]["Forager"]))

    candidates = find_trophallaxis_candidates("data/colony_A.avi", "data/raw_fluorescence_colony_A.csv", colony_A_events, 42, foragers_A)
    candidate_list = []
    for candidate in candidates:
        candidate_list.append([candidate.ants[0], candidate.ants[1], candidate.initial_time,
                               candidate.final_time, len(candidate.event_list), candidate.event_list])
    candidate_df = pd.DataFrame(candidate_list, columns=["ant1", "ant2", "initial_time",
                                                         "final_time", "num_frames", "frames"])
    candidate_df.to_csv("data/A_candidates.csv")

    # colony_B_events = trophallaxis_df[trophallaxis_df["Experiment"] == "B"]
    # draw_ant_boxes("data/colony_B.avi", "data/raw_fluorescence_colony_B.csv", colony_B_events)

    # colony_C_events = trophallaxis_df[trophallaxis_df["Experiment"] == "C"]
    # draw_ant_boxes("data/colony_C.avi", "data/raw_fluorescence_colony_C.csv", colony_C_events)

    stop_time = timeit.default_timer()
    time_elapsed = (stop_time - start_time) / 60.0
    print("Processing took ", time_elapsed, " minutes")