import os
import h5py
import math
import numpy as  np

from structures import Intention
from models import EDGE_DIST_CLASSES
from tqdm import tqdm

import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class DataSampler():

    def __init__(
        self, 
        depth_stack_size=10, 
        depth_sample_period_secs=0.5, 
        max_sample_count_per_behaviour=None,
        behaviour_sample_fraction=0.75,
        use_changepoint_labels=True,
        edge_class_dist_ranges=EDGE_DIST_CLASSES,
        keep_all_data=False,
        only_counting=False
        ):
        self.depth_stack_size = depth_stack_size
        self.depth_sample_period_secs = depth_sample_period_secs
        self.max_sample_count_per_behaviour = max_sample_count_per_behaviour
        self.behaviour_sample_fraction = behaviour_sample_fraction
        self.use_cp_labels = use_changepoint_labels
        self.edge_class_dist_ranges = edge_class_dist_ranges
        self.only_counting = only_counting
        self.keep_all_data = keep_all_data


    def _find_nearest(self, value, buffer):
        idx = np.searchsorted(buffer, value, side="left")
        if idx > 0 and (idx == len(buffer) or math.fabs(value - buffer[idx-1]) < math.fabs(value - buffer[idx])):
            return idx-1
        else:
            return idx

    def _is_valid_sample(self, curr_idx, curr_label, curr_timestamp, depth_timestamps):
        if curr_label == -1:
            return None

        buffer_start_timestamp = depth_timestamps[0]
        offsets = [
            curr_timestamp - self.depth_sample_period_secs * (i+1)
            for i in range(self.depth_stack_size - 1)
            ]
        sampled_img_idxs = [curr_idx]
        for offset in offsets:
            if offset < buffer_start_timestamp:
                return None
            else:
                idx = self._find_nearest(offset, depth_timestamps)
                sampled_img_idxs.append(idx)
        return sampled_img_idxs

    def _process_file_valid_samples(self, h5_path, area_idx, file_idx):
        with h5py.File(h5_path) as f:
            segment_count = int(np.array(f['segment_count']))
            depth_timestamps = np.array(f['depth_timestamps'])
            depth_int_labels = np.array(f['depth_edge_int_labels'])
            depth_edge_idxs = np.array(f['depth_edge_idxs'])
            depth_segment_idxs = np.array(f['depth_segment_idxs'])
            depth_positions = np.array(f['depth_positions'])
            changepoint_poses = np.array(f['changepoint_poses'])

            if self.use_cp_labels:
                depth_changepoint_labels = np.array(f['depth_changepoint_labels'])
            else:
                depth_changepoint_labels = []
                for seg_idx, pos in zip(depth_segment_idxs, depth_positions):
                    if seg_idx < len(changepoint_poses):
                        cp_pos = changepoint_poses[seg_idx]
                        dist = np.linalg.norm(cp_pos - pos)
                    else:
                        dist = np.inf

                    found = False
                    for idx, (start, end) in enumerate(self.edge_class_dist_ranges):
                        if start <= dist and dist < end:
                            depth_changepoint_labels.append(idx)
                            found = True
                            break
                        elif dist == np.inf:
                            depth_changepoint_labels.append(len(self.edge_class_dist_ranges)-1)
                            found = True
                            break

                    if not found:
                        raise Exception("Dist ", dist, " not in any interval!")
                            
            if segment_count == 0:
                return None

            valid_samples = []

            for idx, (ts, int_label, edge_idx, segment_idx, changepoint_label) in enumerate(
                zip(depth_timestamps, depth_int_labels, depth_edge_idxs, depth_segment_idxs, depth_changepoint_labels)
                ):

                sampled_img_idxs = self._is_valid_sample(idx, int_label, ts, depth_timestamps)
                if sampled_img_idxs is not None:
                    valid_samples.append((
                        sampled_img_idxs, # Selected img idxs
                        int_label, # Edge intention label
                        edge_idx, # Associated edge idx
                        changepoint_label, # Changepoint label
                        [area_idx, file_idx] # File descriptor
                    ))

            fig, ax = plt.subplots(1, 1)
            pose_colours = ['c' if label == 0 else 'k' for label in depth_changepoint_labels]
            ax.scatter(depth_positions[:, 0], depth_positions[:, 1], c=pose_colours)
            ax.scatter(changepoint_poses[:, 0], changepoint_poses[:, 1], marker='x')
            ax.scatter(depth_positions[0, 0], depth_positions[0, 1], c='r', marker='o')
            ax.scatter(depth_positions[-1, 0], depth_positions[-1, 1], c='g', marker='o')
            ax.set_aspect('equal')
            plt.savefig('dist_ims/' + str(area_idx) + '_' + str(file_idx) + '.png')
            plt.close()
            
            return valid_samples
    
    def process_dataset(self, data_dir):
        # First scan through the data to see how many valid instances we have
        behaviour_counts = [0 for i in range(len(Intention))]
        area_keys = [
            directory for directory in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, directory))
            ]
        file_keys = []
        valid_samples = []

        for area_idx, area_dir in enumerate(area_keys):
            area_path = os.path.join(data_dir, area_dir)
            area_files = [file for file in os.listdir(area_path) if file.endswith('.h5')]
            file_keys.append(area_files)

            for file_idx, file in enumerate(area_files):
                print(file)
                h5_path = os.path.join(area_path, file)
                file_valid_samples = self._process_file_valid_samples(h5_path, area_idx, file_idx)
                curr_file_valid_labels = [label for _, label, _, _, _ in file_valid_samples]
                for i in range(len(behaviour_counts)):
                    behaviour_counts[i] += np.count_nonzero(np.array(curr_file_valid_labels) == i)
                valid_samples += file_valid_samples

        print(behaviour_counts)

        if self.only_counting:
            return

        # Process to extract valid data instances, then sample to balance data across behaviours
        if self.keep_all_data:
            samples = [
                list(reversed([i for i in range(behaviour_counts[idx])]))
                for idx in range(len(behaviour_counts))
                ]

            print("Keeping all data")
        else:
            min_idx = np.argmin(behaviour_counts)
            min_count = behaviour_counts[min_idx]
            min_count = (
                min(self.max_sample_count_per_behaviour, min_count)
                if self.max_sample_count_per_behaviour is not None else min_count
                )
            min_count = int(np.floor(min_count * self.behaviour_sample_fraction))
            print("Sampling ", min_count, " from each behaviour")

            samples = [
                np.sort(np.random.choice(behaviour_counts[idx], min_count, replace=False))[::-1].tolist()
                for idx in range(len(behaviour_counts))
                ]

        behaviour_idxs = [0 for i in range(len(Intention))]

        # Store indices to allow us to find the specific point (area, file, idx in file)
        data_img_idxs = []
        data_edge_int_labels = []
        data_edge_idxs = []
        data_file_descs = []
        data_changepoint_labels = []

        for img_idxs, int_label, edge_idx, changepoint_label, file_desc in valid_samples:
            curr_behaviour_idx = behaviour_idxs[int_label]

            if len(samples[int_label]) > 0 and samples[int_label][-1] == curr_behaviour_idx:
                samples[int_label].pop()
                data_img_idxs.append(img_idxs)
                data_edge_int_labels.append(int_label)
                data_edge_idxs.append(edge_idx)
                data_file_descs.append(file_desc)
                data_changepoint_labels.append(changepoint_label)

            behaviour_idxs[int_label] += 1

        out_file = os.path.join(data_dir, 'ros_formatted.npz')
        np.savez(
            out_file,
            area_keys=area_keys,
            file_keys=file_keys,
            data_img_idxs=np.array(data_img_idxs),
            data_edge_int_labels=data_edge_int_labels,
            data_changepoint_labels = data_changepoint_labels,
            data_edge_idxs=data_edge_idxs,
            data_file_descs=np.array(data_file_descs),
            depth_stack_size=self.depth_stack_size,
            depth_sample_period_secs=self.depth_sample_period_secs,
        )


if __name__ == "__main__":
    # data_dir = "/data/home/joel/datasets/blocal_data/test_h5"
    # data_dir = "/Users/joel/Research/data/ros_test/test_h5_v2"
    # data_dir = "/data/home/joel/datasets/blocal_data/blocal_h5_data/"
    # data_dir = "/data/home/joel/datasets/blocal_data/blocal_h5_smallsize_cps"
    # data_dir = "/data/home/joel/datasets/blocal_data/com1_basement_test"
    # data_dir = "/data/home/joel/datasets/blocal_data/blocal_odom_h5_train"
    data_dir = "/data/home/joel/datasets/blocal_data/blocal_odom_h5_test"

    # sampler = DataSampler(keep_all_data=True)
    # sampler = DataSampler()
    sampler = DataSampler(
        behaviour_sample_fraction=0.92,
        # keep_all_data=True,
        use_changepoint_labels=False
    )
    sampler.process_dataset(data_dir)
