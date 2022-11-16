import os
import cv2
import h5py
import numpy as np
from collections import deque
from rosbag import Bag
from cv_bridge import CvBridge
from behaviour_msgs.msg import GraphLocationStamped


class BagProcessor():
    def __init__(self):
        self.DEPTH_IMAGE_TIMEOUT_SECS = 0.08
        self.bridge = CvBridge()
        self.img_size = (212, 120)

    def _get_synchronized_depth_image(self, cameras):
        # Assumes that all cameras have at least one image buffered
        synced = [len(depth_ims) - 1 for depth_ims in cameras]
        timestamps = [cameras[cam][idx].header.stamp.to_sec() for cam, idx in enumerate(synced)]
        max_ts = max(timestamps)
        min_ts = min(timestamps)

        # Try to find a matching triplet of images, by removing 
        # the image with the largest timestamp at each iteration
        while max_ts - min_ts > self.DEPTH_IMAGE_TIMEOUT_SECS:
            max_cam = np.argmax(timestamps)

            if synced[max_cam] == 0:
                # No matching images found within the buffer
                return None

            synced[max_cam] -= 1
            timestamps[max_cam] = cameras[max_cam][synced[max_cam]].header.stamp.to_sec()
            max_ts = max(timestamps)
            min_ts = min(timestamps)

        # Gather all the synced images
        synced_ims = [
            cv2.resize(
                self.bridge.imgmsg_to_cv2(cameras[cam][idx], desired_encoding='passthrough'),
                (212, 120),
                interpolation=cv2.INTER_NEAREST
            )
            for cam, idx in enumerate(synced)
        ]
        
        # Remove all the dated images up to the images used for the current synced image
        for i in range(len(cameras)):
            for _ in range(synced[i] + 1):
                cameras[i].popleft()

        return min_ts, synced_ims

    def process_bag_to_h5py(self, bagfile, outfile):
        # Parse bagfile first
        msgs = []
        with Bag(bagfile) as bag:
            for topic, msg, _ in bag.read_messages():
                msgs.append([topic, msg])
        
        print("Re-ordering the messages by header timestamp...")
        reordered_msgs = sorted(msgs, key=lambda topic_msg: topic_msg[1].header.stamp.to_sec())
        start_time = reordered_msgs[0][1].header.stamp.to_sec()
        end_time = reordered_msgs[-1][1].header.stamp.to_sec()

        print("Synchronizing images...")
        combined_depth_ims = []
        combined_depth_stamps = []
        segments = [(None, None, start_time)]
        depth_ims = [deque(maxlen=5) for i in range(3)]
        for topic, msg in reordered_msgs:
            if 'left/depth' in topic:
                depth_ims[0].append(msg)
            elif 'mid/depth' in topic:
                depth_ims[1].append(msg)
            elif 'right/depth' in topic:
                depth_ims[2].append(msg)
            
            if all([len(dq) > 0 for dq in depth_ims]):
                min_ts, synced_ims = self._get_synchronized_depth_image(depth_ims)
                if synced_ims is not None:
                    combined_depth_ims.append(synced_ims)
                    combined_depth_stamps.append(min_ts)

            if topic == '/graph_location':
                _, prev_edge_idx, _ = segments[-1]
                if msg.edge_idx != prev_edge_idx:
                    segments.append((msg.intention.val, msg.edge_idx, msg.header.stamp.to_sec()))
            
        print("Matching combined depth images to behaviours...")
        segment_end_times = [segments[idx][2] for idx in  range(1, len(segments))] + [end_time]
        segment_idx = 0
        segment_lens = [0]

        depth_labels = []
        depth_edge_idxs = []

        for idx, ts in enumerate(combined_depth_stamps):
            if ts > segment_end_times[segment_idx]:
                segment_lens.append(1)
                segment_idx += 1
            else:
                segment_lens[-1] += 1

            label, edge_idx, _ = segments[segment_idx]
            label = -1 if label is None else label
            edge_idx = -1 if edge_idx is None else edge_idx
            depth_labels.append(label)
            depth_edge_idxs.append(edge_idx)


        segment_cumlens = np.cumsum(np.array(segment_lens))

        # Write data into h5py file
        print("Writing data")
        with h5py.File(outfile, 'w') as hf:
            hf.create_dataset('end_time', data=end_time)
            hf.create_dataset('segment_count', data=len(segment_lens))
            hf.create_dataset('segment_cumlen', data=segment_cumlens)
            hf.create_dataset('combined_depth', data=combined_depth_ims, compression="gzip")
            hf.create_dataset('depth_labels', data=depth_labels)
            hf.create_dataset('depth_edge_idxs', data=depth_edge_idxs)
            hf.create_dataset('depth_timestamps', data=combined_depth_stamps)
            for idx, segment in enumerate(segments):
                if segment[0] is None:
                    hf.create_dataset(str(idx) + '/label', data=-1)
                    hf.create_dataset(str(idx) + '/edge_idx', data=-1)
                    hf.create_dataset(str(idx) + '/start_time', data=segment[2])
                else:
                    hf.create_dataset(str(idx) + '/label', data=segment[0])
                    hf.create_dataset(str(idx) + '/edge_idx', data=segment[1])
                    hf.create_dataset(str(idx) + '/start_time', data=segment[2])


if __name__ == "__main__":
    proc = BagProcessor()
    data_dir = '/home/files/test'
    out_dir = '/home/files/test_h5'

    for area_dir in os.listdir(data_dir):
        area_path = os.path.join(data_dir, area_dir)
        if not os.path.exists(os.path.join(out_dir, area_dir)):
            os.mkdir(os.path.join(out_dir, area_dir))

        for file in os.listdir(area_path):
            filepath = os.path.join(area_path, file)
            filename = os.path.splitext(file)[0]
            writepath = os.path.join(out_dir, area_dir, filename + '.h5')
            print(">>> Processing ", filepath)
            proc.process_bag_to_h5py(filepath, writepath)