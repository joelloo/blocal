import os
import sys
import glob
import h5py
import numpy as  np

from structures import Intention
from tqdm import tqdm

if __name__ == "__main__":
    # data_dir = "/Users/joel/Research/data/test"
    # output_dir = "/Users/joel/Research/data/"
    data_dir = "/data/home/joel/datasets/blocal_data_x3"
    output_dir = "/data/home/joel/datasets/blocal_data_formatted"
    depth_stack_size = 20
    only_counting = False
    max_sample_count_per_behaviour = None

    # First scan through the data to see how many valid instances we have
    behaviour_counts = [0 for i in range(len(Intention))]

    for directory in os.listdir(data_dir): 
        for filename in tqdm(os.listdir(os.path.join(data_dir, directory))):
            if filename.endswith(".h5"):
                with h5py.File(os.path.join(data_dir, directory, filename)) as f:
                    segment_count = int(np.array(f['segment_count']))
                    segment_cumlen = f['segment_cumlen']
                    segment_cumlen_padded = np.insert(segment_cumlen, 0, 0)
                    
                    has_init_images = False
                    for i in range(segment_count):
                        if not has_init_images and segment_cumlen_padded[i+1] >= depth_stack_size:
                            has_init_images = True
                            segment_int = Intention(np.array(f[str(i) + '/label']))
                            behaviour_counts[int(segment_int)] += segment_cumlen_padded[i+1] - depth_stack_size + 1
                            continue

                        if has_init_images:
                            segment = f[str(i)]
                            segment_int = Intention(np.array(segment['label']))
                            behaviour_counts[int(segment_int)] += segment_cumlen_padded[i+1] - segment_cumlen_padded[i]
    
    print([(Intention(i), behaviour_counts[i]) for i in range(len(Intention))])
    if only_counting:
        sys.exit(0)

    # Process to extract valid data instances, then sample to balance data across behaviours
    min_idx = np.argmin(behaviour_counts)
    min_count = behaviour_counts[min_idx]
    min_count = (
        min(max_sample_count_per_behaviour, min_count) 
        if max_sample_count_per_behaviour is not None else min_count
        )

    samples = [
        np.sort(np.random.choice(behaviour_counts[idx], min_count, replace=False))[::-1].tolist()
        for idx in range(len(behaviour_counts))
        ]
    behaviour_idxs = [0 for i in range(len(Intention))]

    # print(">>>>>")
    # print(samples)
    # print("<<<<<")

    dataset_keys = os.listdir(data_dir)
    file_keys = [
        [f for f in os.listdir(os.path.join(data_dir, directory)) if f.endswith('.h5')]
        for directory in dataset_keys
        ]

    # Store indices to allow us to find the specific point (area, file, segment, point)
    # Also store the index to find the start of the depth image stack in the stored
    # collection of depth images, as well as the index of the current edge.
    data_idxs = []

    for dir_idx, (directory, dataset_files) in enumerate(zip(dataset_keys, file_keys)):
        dataset_dir = os.path.join(data_dir, directory)

        for file_idx, filename in tqdm(enumerate(dataset_files)):
            with h5py.File(os.path.join(dataset_dir, filename)) as f:
                segment_count = int(np.array(f['segment_count']))
                segment_cumlen = f['segment_cumlen']
                segment_cumlen_padded = np.insert(segment_cumlen, 0, 0)

                for i in range(segment_count):
                    segment_int = Intention(np.array(f[str(i) + '/label']))
                    segment_edge = int(np.array(f[str(i) + '/edge_idx']))
                    for im_idx in range(segment_cumlen_padded[i], segment_cumlen_padded[i+1]):
                        if im_idx < depth_stack_size - 1:
                            continue

                        if (
                            len(samples[int(segment_int)]) != 0
                            and behaviour_idxs[int(segment_int)] == samples[int(segment_int)][-1]
                        ):
                            point_idx = im_idx - segment_cumlen_padded[i]

                            # data_idxs is (area idx, file idx, segment idx, point idx -- within the segment, 
                            # index of start of depth image stack for current point,
                            # index of current edge in behaviour graph)
                            data_idxs.append([
                                dir_idx, file_idx, i, point_idx, im_idx + 1 - depth_stack_size, segment_edge
                                ])

                            samples[int(segment_int)].pop()
                        behaviour_idxs[int(segment_int)] += 1

    # print("^^^^^^^^^^^^")
    # print(behaviour_idxs)

    # print("*************")
    # for intention_data in area_traj_idx:
    #     print(len(intention_data), ": ", intention_data)

    out_file = os.path.join(output_dir, 'formatted.npz')
    np.savez(
        out_file, 
        dataset_keys=dataset_keys, 
        file_keys=file_keys, 
        data=data_idxs, 
        depth_stack_size=depth_stack_size
    )

