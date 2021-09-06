import pandas as pd
import numpy as np
import timeit

def calculate_crop_differences(ant_data, candidates):

    #candidates["num_frames"] = candidates["frames"].apply(lambda x: len(list(x)))

    for index, row in candidates.iterrows():
        initial_time = row["initial_time"]
        final_time = row["final_time"]
        ant1 = row["ant1"]
        ant2 = row["ant2"]

        ant1_crops = []
        ant2_crops = []

        for i in range(int(initial_time), int(final_time)+1, 2):
            data_row = ant_data.iloc[int(i/2)]
            ant1_crop = data_row[str(ant1) + "-crop_intensity"]
            ant2_crop = data_row[str(ant2) + "-crop_intensity"]
            if ant1_crop == -1:
                ant1_crop = 0
                if len(ant1_crops) > 0:
                    ant1_crop = ant1_crops[-1]

            ant1_crops.append(ant1_crop)

            if ant2_crop == -1:
                ant2_crop = 0
                if len(ant2_crops) > 0:
                    ant2_crop = ant2_crops[-1]

            ant2_crops.append(ant2_crop)

        split_point = int(len(ant1_crops)/2)

        ant1_front = ant1_crops[:split_point]
        if len(ant1_front) > 0:
            ant1_front_mean = np.mean(ant1_front)
        else:
            ant1_front_mean = 0

        ant2_front = ant2_crops[:split_point]
        if len(ant2_front) > 0:
            ant2_front_mean = np.mean(ant2_front)
        else:
            ant2_front_mean = 0

        ant1_back = ant1_crops[split_point:]
        if len(ant1_back) > 0:
            ant1_back_mean = np.mean(ant1_back)
        else:
            ant1_back_mean = 0

        ant2_back = ant2_crops[split_point:]
        if len(ant2_back) > 0:
            ant2_back_mean = np.mean(ant2_back)
        else:
            ant2_back_mean = 0

        candidates.at[index, "difference"] = np.mean([np.abs(ant1_front_mean - ant1_back_mean),
                                                    np.abs(ant2_front_mean - ant2_back_mean)])

    candidates.to_csv("data/A_differences.csv")
    return candidates

########################################################################################################################

def reduce_and_label(candidates, length_cutoff=0, difference_cutoff=0):
    print(len(candidates))
    candidates = candidates[candidates["num_frames"] > length_cutoff]
    print(len(candidates))
    #candidates["label"] = candidates["difference"].apply(lambda x: 1 if x >= difference_cutoff else 0)
    candidates = candidates[candidates["difference"] >= difference_cutoff]
    print(len(candidates))

    #candidates.to_csv("data/A_labelled.csv")
    return candidates

########################################################################################################################

def evaluate_candidates(candidates, true_events, overlap_cutoff=0.8):
    num_matched = 0
    num_extra = 0
    candidates["found"] = -1
    true_events["matched"] = -1
    for index, row in candidates.iterrows():

        # First find all rows with the same ants
        matching = true_events[true_events["ant1"].isin([row["ant1"], row["ant2"]]) |
                               true_events["ant2"].isin([row["ant1"], row["ant2"]])]

        if len(matching) > 0:

            matched = False
            for i, true_row in matching.iterrows():
                candidate_initial = int(row["initial_time"])
                candidate_final = int(row["final_time"])
                true_initial = int(true_row["initial_time"])
                true_final = int(true_row["final_time"])

                candidate_range = set(range(candidate_initial, candidate_final+1, 2))
                true_range = range(true_initial, true_final, 2)

                overlap_range = candidate_range.intersection(true_range)
                if len(overlap_range) / len(candidate_range) > overlap_cutoff or len(overlap_range) / len(true_range) > overlap_cutoff:
                    matched = True
                    candidates.at[index, "found"] = 1
                    true_events.at[i, "matched"] = 0
                    break

            if matched:
                num_matched += 1
                continue
            else:
                num_extra += 1
                continue

        else:
            num_extra += 1

    num_missed = np.abs(np.sum(true_events["matched"]))

    candidates = candidates[candidates["found"] == -1]
    candidates.to_csv("data/predicted_interactions.csv")

    return num_matched, num_extra, num_missed

########################################################################################################################

if __name__ == '__main__':
    start_time = timeit.default_timer()

    ant_data = pd.read_csv("data/raw_fluorescence_colony_A.csv")
    candidates = pd.read_csv("data/A_candidates.csv")
    length_cutoff = 2
    difference_cutoff = 10
    overlap_cutoff = 0.8

    differences = calculate_crop_differences(ant_data, candidates)
    #differences = pd.read_csv("data/A_differences.csv")
    labelled = reduce_and_label(differences, length_cutoff=length_cutoff, difference_cutoff=difference_cutoff)
    #labelled = pd.read_csv("data/A_labelled.csv")

    true_interactions = pd.read_csv("data/interactions_data_A.csv")
    true_interactions = true_interactions[true_interactions["ant1"] > 0]
    true_interactions = true_interactions[true_interactions["ant2"] > 0]
    true_interactions.dropna(inplace=True, subset=["ant1_gave"])
    true_interactions = true_interactions[true_interactions["ant1_gave"] > difference_cutoff]
    print("Number of True Interactions: ", len(true_interactions))

    matched, extra, missed = evaluate_candidates(labelled, true_interactions, overlap_cutoff=overlap_cutoff)
    print("Num Matched: ", matched)
    print("Num Extra: ", extra)
    print("Num Missed: ", missed)

    stop_time = timeit.default_timer()
    time_elapsed = (stop_time - start_time) / 60.0

    print("Processing took ", time_elapsed, " minutes")
