import os
import cv2
import h5py
import numpy as np
from bisect import bisect_left
from collections import deque
from operator import itemgetter
from rosbag import Bag
from cv_bridge import CvBridge
from behaviour_msgs.msg import GraphLocationStamped
# from tf.transformations import translation_matrix, quaternion_matrix, quaternion_slerp
from transformations import translation_matrix, quaternion_matrix, quaternion_slerp

# start is inclusive, end is exclusive and start < end
def list_rindex(l, start, end, elem):
    for idx in range(end-1, start-1, -1):
        if l[idx] == elem:
            return idx
    return ValueError(str(elem) + " is not in list")


class BagProcessor():
    def __init__(
        self, 
        use_changepoint_labels=False,
        junction_buffer_dist=1.0
        ):

        self.DEPTH_IMAGE_TIMEOUT_SECS = 0.08
        self.bridge = CvBridge()
        self.img_size = (212, 120)
        self.pose_buffer = None
        self.pose_timestamps = None
        self.use_cp_labels = use_changepoint_labels

        self.junction_buffer_dist = junction_buffer_dist

    def _get_interpolated_pose(self, timestamp):
        # i = bisect_left(self.pose_buffer, timestamp, key=itemgetter(0))
        i = bisect_left(self.pose_timestamps, timestamp)
        if 0 < i and i < len(self.pose_buffer):
            # Requested timestamp falls inside pose buffer, can interpolate pose.
            sstamp, spos, squat = self.pose_buffer[i-1]
            estamp, epos, equat = self.pose_buffer[i]

            inv_period = 1 / (estamp - sstamp)
            sdiff = (timestamp - sstamp) * inv_period
            ediff = (estamp - timestamp) * inv_period

            slerped_quat = quaternion_slerp(squat, equat, sdiff)
            R = quaternion_matrix(slerped_quat)

            lerped_pos = sdiff * epos + ediff * spos

            interp_pose = R
            interp_pose[:-1, -1] = lerped_pos

            return interp_pose

        else:
            print("No valid pose for: ", timestamp)
            return None

    def _search_dist(self, start, end, dist_threshold, buf):
        incr = 1 if start < end else -1
        init_pose, _ = buf[start]
        init_position = init_pose[:-1, -1]

        for i in range(start, end, incr):
            pose, _ = buf[i]
            dist = np.linalg.norm(init_position - pose[:-1, -1])
            if dist > dist_threshold:
                return i

        return None

    def _search_backward(self, start, end, buf):
        return self._search_dist(end, start-1, self.junction_buffer_dist, buf)

    def _search_forward(self, start, end, buf):
        return self._search_dist(start, end+1, self.junction_buffer_dist, buf)

    def _extract_behaviour_segments(self, graph_poses, traj_end_time):
        if len(graph_poses) == 0:
            return None

        recorded_segments = []
        segment_edge_idxs = []
        segment_edge_ints = []
        start = 0
        curr_edge_idx = graph_poses[0][1].edge_idx
        curr_edge_int = graph_poses[0][1].intention.val

        for idx, (_, msg) in enumerate(graph_poses):
            if msg.edge_idx != curr_edge_idx:
                recorded_segments.append((start, idx))
                segment_edge_idxs.append(curr_edge_idx)
                segment_edge_ints.append(curr_edge_int)
                start = idx
                curr_edge_idx = msg.edge_idx
                curr_edge_int = msg.intention.val
        recorded_segments.append((start, len(graph_poses)))
        segment_edge_idxs.append(curr_edge_idx)
        segment_edge_ints.append(curr_edge_int)

        junction_segments = []
        junction_marks = [msg.junction for _, msg in graph_poses]

        for start, end in recorded_segments:
            try:
                junction_start = junction_marks.index(True, start, end)
                junction_end = list_rindex(junction_marks, start, end, True)
                junction_segments.append((junction_start, junction_end))
            except ValueError:
                junction_segments.append((None, None))

        print("@@@ ", recorded_segments)
        print(">>> ", junction_segments, len(graph_poses))
        
        junction_poses = []
        for p1, p2 in junction_segments:
            if p1 is not None:
                junction_poses.append(graph_poses[p1][0][:-1, -1])
                junction_poses.append(graph_poses[p2][0][:-1, -1])

        # Temporarily include array's first element in changepoint_idxs,
        # for easy computation in the for loop
        changepoint_idxs = [0]

        for i in range(len(recorded_segments) - 1):
            curr_seg, next_seg = recorded_segments[i], recorded_segments[i+1]
            curr_juncs, next_juncs = junction_segments[i], junction_segments[i+1]

            if curr_juncs[1] is None and next_juncs[0] is None:
                raise Exception("Two consecutive segments with no junctions!")
            elif curr_juncs[1] is None and next_juncs[0] is not None:
                cp_idx = self._search_backward(
                    min(changepoint_idxs[-1], curr_seg[0]), 
                    next_juncs[0], graph_poses
                )
            elif curr_juncs[1] is not None and next_juncs[0] is None:
                cp_idx = self._search_forward(
                    curr_juncs[1], next_seg[1], graph_poses
                )
            else: # curr_juncs[1] is not None and next_juncs[0] is not None
                bottom_cp_idx = self._search_forward(
                    curr_juncs[1], next_juncs[0], graph_poses
                )
                top_cp_idx = self._search_backward(
                    curr_juncs[1], next_juncs[0], graph_poses
                )
                cp_idx = min(top_cp_idx, bottom_cp_idx)

            changepoint_idxs.append(cp_idx)

        changepoint_idxs = changepoint_idxs[1:] # Remove the temporarily placed first element
        changepoint_poses = [
            graph_poses[idx][0][:-1, -1] for idx in changepoint_idxs
        ]
        segment_end_times = [
            graph_poses[i][1].header.stamp.to_sec() for i in changepoint_idxs
        ]
        segment_end_times.append(traj_end_time)
        segments = list(zip(segment_edge_ints, segment_edge_idxs, segment_end_times))

        print("&&& ", changepoint_idxs)

        return changepoint_idxs, changepoint_poses, junction_poses, segments

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
        
        # Get the pose of the 'synced' images, using the images'
        # earliest timestamp. If there is no valid pose, return None.
        interp_pose = self._get_interpolated_pose(min_ts)
        if interp_pose is None:
            return None

        # Remove all the dated images up to the images used for the current synced image
        for i in range(len(cameras)):
            for _ in range(synced[i] + 1):
                cameras[i].popleft()

        return min_ts, synced_ims, interp_pose

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

        print("Collect poses first...")
        self.pose_buffer = []
        for topic, msg in reordered_msgs:
            if 'spot/odometry' in topic:
                pos = msg.pose.pose.position
                quat = msg.pose.pose.orientation

                self.pose_buffer.append((
                    msg.header.stamp.to_sec(),
                    np.array([pos.x, pos.y, pos.z]), 
                    np.array([quat.x, quat.y, quat.z, quat.w])
                ))
        self.pose_timestamps = [ts for ts, _, _ in self.pose_buffer]

        print("Synchronizing images...")
        combined_depth_ims = []
        combined_depth_stamps = []
        combined_depth_poses = []
        graph_poses = []
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
                ret = self._get_synchronized_depth_image(depth_ims)
                if ret is not None:
                    min_ts, synced_ims, pose = ret
                    combined_depth_ims.append(synced_ims)
                    combined_depth_stamps.append(min_ts)
                    combined_depth_poses.append(pose)

            if topic == '/graph_location':
                if self.use_cp_labels:
                    _, prev_edge_idx, _ = segments[-1]
                    if msg.edge_idx != prev_edge_idx:
                        segments.append((msg.intention.val, msg.edge_idx, msg.header.stamp.to_sec()))
                else:
                    pose = self._get_interpolated_pose(msg.header.stamp.to_sec())
                    if pose is not None:
                        graph_poses.append((pose, msg))

        # intentions = np.array([msg.intention.val for pose, msg in graph_poses])
        # eidxs = np.array([msg.edge_idx for pose, msg in graph_poses])
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # print(intentions)
        # np.set_printoptions(threshold=10)

        print("Matching combined depth images to behaviours...")
        if not self.use_cp_labels:
            _, cp_poses, jn_poses, segments = self._extract_behaviour_segments(graph_poses, end_time)
            changepoint_poses = np.array(cp_poses)
            segment_end_times = [end_time for _, _, end_time in segments]
        else:
            changepoint_poses = np.array([])
            segment_end_times = [segments[idx][2] for idx in  range(1, len(segments))] + [end_time]

        segment_idx = 0
        segment_lens = [0]
        depth_positions = []
        depth_edge_ints = []
        depth_edge_idxs = []

        depth_ts_poses = zip(combined_depth_stamps, combined_depth_poses)
        for idx, (ts, pose) in enumerate(depth_ts_poses):
            if ts > segment_end_times[segment_idx]:
                segment_lens.append(1)
                segment_idx += 1
            else:
                segment_lens[-1] += 1

            label, edge_idx, _ = segments[segment_idx]
            label = -1 if label is None else label
            edge_idx = -1 if edge_idx is None else edge_idx
            depth_edge_ints.append(label)
            depth_edge_idxs.append(edge_idx)
            depth_positions.append(pose[:-1, -1])

        segment_cumlens = np.cumsum(np.array(segment_lens))
        depth_segment_idxs = [
            i for i, seg_len in enumerate(segment_lens) for _ in range(seg_len)
        ]

        # Write data into h5py file
        print("Writing data")
        with h5py.File(outfile, 'w') as hf:
            hf.create_dataset('end_time', data=end_time)
            hf.create_dataset('segment_count', data=len(segment_lens))
            hf.create_dataset('segment_cumlen', data=segment_cumlens)
            hf.create_dataset('changepoint_poses', data=changepoint_poses)
            hf.create_dataset('combined_depth', data=combined_depth_ims, compression="gzip")
            hf.create_dataset('depth_positions', data=depth_positions)
            hf.create_dataset('depth_edge_int_labels', data=depth_edge_ints)
            hf.create_dataset('depth_edge_idxs', data=depth_edge_idxs)
            hf.create_dataset('depth_segment_idxs', data=depth_segment_idxs)
            hf.create_dataset('depth_timestamps', data=combined_depth_stamps)
            hf.create_dataset('junction_poses', data=jn_poses)
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
    data_dir = '/home/files/blocal_odom_ros'
    out_dir = '/home/files/blocal_odom_h5'

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