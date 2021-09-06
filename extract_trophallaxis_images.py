import matplotlib
#matplotlib.use("TKAGG")
#import matplotlib.pyplot as plt
import timeit
import pandas as pd
import cv2
import numpy as np

########################################################################################################################
""" Take in a video file and csv file, and plot the bounding boxes around the corresponding ants as recorded
    in the csv.
    :returns an edited video"""

def crop_ant_boxes(video_file, csv_file, trophallaxis_df, queen_number, save_directory):
    # Hyperparameters
    search_radius = 30

    # Frames are 1790 x 1200
    test_video = cv2.VideoCapture(video_file)
    good_read, frame = test_video.read()
    if not good_read:
        print("Error getting frame size from ", video_file)

    video = cv2.VideoCapture(video_file)
    writer = cv2.VideoWriter(video_file[:-4] + "_annotated.avi",
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
                             (frame.shape[1], frame.shape[0]))

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

        id = 0
        ant_heads = {}
        # Get the current frame to mark up with boxes
        good_read, frame = video.read()
        if not good_read:
            print("Error reading frame ", count, " from video file ", video_file)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
            head_points.append(np.array([head_x, head_y]))
            ant_heads[int(ant)] = (head_x, head_y)

        # Compare all points, redraw circle for candidate trophallaxis events
        for point in head_points:
            for other_point in head_points:
                if point[0] == other_point[0] and point[1] == other_point[1]:
                    continue
                else:
                    distance = np.linalg.norm(point-other_point)
                    if distance < (search_radius * 2) - 1:
                        pass
                        #cv2.circle(frame, (point[0], point[1]), search_radius, (0, 255, 0), 2)

        # Check if there were any trophallaxis events at this frame

        # Radius parameter to control image size.
        radius = 50
        side_length = radius*2
        events = trophallaxis_df.loc[trophallaxis_df["final_time"] >= frame_idx]
        events = events.loc[events["initial_time"] <= frame_idx]
        for index, row in events.iterrows():
            contour_list = []
            seed_ant = int(row["ant1"])
            if seed_ant == -1:
                seed_ant = int(row["ant2"])
                if seed_ant == -1:
                    continue
            seed_ant_center = ant_heads[seed_ant]
            if seed_ant_center[0] == -1 or seed_ant_center[1] == -1:
                seed_ant = int(row["ant2"])
                if seed_ant > 0:
                    ant_center = ant_heads[seed_ant]
                    if seed_ant_center[0] == -1 or seed_ant_center[1] == -1:
                        continue
                else:
                    continue

            # Take the bounding box around the identified ants' head
            bounding_rect = [seed_ant_center[0]-radius, seed_ant_center[1]-radius, side_length, side_length]

            #cv2.rectangle(frame, (int(bounding_rect[0]), int(bounding_rect[1])),
            #              (int(bounding_rect[0])+int(bounding_rect[2]),  int(bounding_rect[1]) + int(bounding_rect[3])),
            #              (0, 255, 255), 5)

            try:
                crop = frame[bounding_rect[1]:bounding_rect[1]+bounding_rect[3], bounding_rect[0]:bounding_rect[0]+bounding_rect[2]]
                crop = cv2.resize(crop, (side_length, side_length))
                cv2.imwrite(save_directory + str(frame_idx) + "-" + str(id) + ".jpg", crop)
                id += 1
            except:
                print("Error extracting from frame ", frame_idx)

            # plt.imshow(frame)
            # plt.savefig("colony_A_bb.jpg")
            # plt.show()

        writer.write(frame)


        if (count+1) % 1000 == 0:
            print("Processed " + str(count+1) + " frames")
        count += 1

    writer.release()
    return

########################################################################################################################

if __name__ == '__main__':
    start_time = timeit.default_timer()

    colony_A_events = pd.read_csv("data/interactions_data_A.csv")
    colony_A_nonevents = pd.read_csv("data/predicted_interactions.csv")
    crop_ant_boxes("data/colony_A.avi", "data/raw_fluorescence_colony_A.csv", colony_A_events, 42, "images/truths/A-")
    crop_ant_boxes("data/colony_A.avi", "data/raw_fluorescence_colony_A.csv", colony_A_nonevents, 42, "images/falses/A-")

    # colony_B_events = trophallaxis_df[trophallaxis_df["Experiment"] == "B"]
    # draw_ant_boxes("data/colony_B.avi", "data/raw_fluorescence_colony_B.csv", colony_B_events)

    # colony_C_events = trophallaxis_df[trophallaxis_df["Experiment"] == "C"]
    # draw_ant_boxes("data/colony_C.avi", "data/raw_fluorescence_colony_C.csv", colony_C_events)

    stop_time = timeit.default_timer()
    time_elapsed = (stop_time - start_time) / 60.0
    print("Processing took ", time_elapsed, " minutes")