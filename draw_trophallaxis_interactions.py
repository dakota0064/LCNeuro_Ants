import matplotlib
matplotlib.use("TKAGG")
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

########################################################################################################################
""" Take in a video file and csv file, and plot the bounding boxes around the corresponding ants as recorded
    in the csv.
    :returns an edited video"""

def draw_ant_boxes(video_file, csv_file):
    video = cv2.VideoCapture(video_file)
    #writer = cv2.VideoWriter(video_file + "_annotated", 60, )
    df = pd.read_csv(csv_file)
    max_index = len(df["time[s] #"])
    count = 0

    column_headers = list(df.columns.values)
    ant_indices = set()

    # Get the indices of the individual ants, since they will each have a column header with a '-x' value
    for header in column_headers:
        if ("-x") in header:
            ant_indices.add(header[:-2])

    ant_indices = list(ant_indices)
    ant_indices.sort()

    while(count < max_index):

        # Get the current frame to mark up with boxes
        flag, frame = video.read()
        if not flag:
            print("Error reading frame ", count, " from video file ", video_file)

        print(frame.shape)

        row = df.iloc[count]
        for index in ant_indices:
            x = row[index + "-x"]
            y = row[index + "-y"]
            angle = row[index + "-angle"]

            print(x, y, angle)

            # If any -1 values, we have incomplete data for this ant at this frame
            if x == -1 or y == -1 or angle == -1:
                continue

            # Ants appear approx 100px long and 30px wide
            #Todo find a better way to compute this

            # rect = ((center_pt), (width, height), angle)
            rect = ((x, y), (100, 30), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw the box over the corresponding ant
            cv2.drawContours(frame, [box], 0, (255, 255, 0), 2)



        plt.imshow(frame)
        plt.savefig("colony_A_bb.jpg")
        plt.show()

        count += 1

########################################################################################################################

if __name__ == '__main__':
    draw_ant_boxes("data/colony_A.avi", "data/raw_fluorescence_colony_A.csv")