import os
import cv2
import sys
import h5py
import json
import argparse
import numpy as np

from structures import BehaviourGraph
from map_utils import mapPoint2Pixel, convertMapFrame

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from vis_utils import tile_figures

matplotlib.rc('image', cmap='jet')


image_pointer = 0

def on_press(event):
    print('Received: ', event.key)

    global image_pointer
    if event.key == 'right':
        plt.close('all')
        image_pointer += 10
    elif event.key == 'up':
        plt.close('all')
        image_pointer += 1
    elif event.key == 'down':
        plt.close('all')
        image_pointer -= 1
    elif event.key == 'left':
        plt.close('all')
        image_pointer -= 10

# graph_path = '/Users/joel/Downloads/catkin_ws/src/graph_localizer/src/COM1_L1_graph.json'
# results_path = '/Users/joel/Research/behaviour_mapping/blocal/results_ros_v2.json'

# graph_path = '/Users/joel/Downloads/catkin_ws/src/graph_localizer/src/COM1_Basement_graph.json'
# results_path = '/Users/joel/Research/behaviour_mapping/blocal/results_ros_com1_basement.json'

graph_path = '/Users/joel/Research/data/ros_test/blocal_odom_h5_test/COM1_L1/behaviour_graph.json'
results_path = '/Users/joel/Research/behaviour_mapping/blocal/results.json'

graph = BehaviourGraph()
graph.loadGraph(graph_path)

map_img = cv2.imread('/Users/joel/Downloads/catkin_ws/src/graph_localizer/src/COM1_L1.jpeg')
# map_img = cv2.imread('/Users/joel/Downloads/catkin_ws/src/graph_localizer/src/COM1_Basement.jpeg')
map_img = map_img[:, :, ::-1] # Reverse RGB channel to display CV images correctly
map = convertMapFrame(map_img)

# tmp = np.load('/Users/joel/Research/data/ros_test/blocal_h5_smallsize_val/ros_formatted.npz')
# tmp = np.load('/Users/joel/Research/data/ros_test/com1_basement_test/ros_formatted.npz')
tmp = np.load('/Users/joel/Research/data/ros_test/blocal_odom_h5_test/ros_formatted.npz')
data_file_descs = tmp['data_file_descs']
data_img_idxs = tmp['data_img_idxs']
area_keys = tmp['area_keys']
file_keys = tmp['file_keys']

results = None
with open(results_path, 'r') as f:
    results = json.load(f)


combined = list(zip(results['targets'], results['subgraph_edge_output'], results['dist_probs']))

# for idx, (target, subgraph, dist_probs) in enumerate(zipped_results):
#     if idx % 10 != 0:
#         continue

while True:
    target, subgraph, dist_probs = combined[image_pointer]

    fig, axs = plt.subplots(2, 1)
    fig.canvas.mpl_connect('key_press_event', on_press)

    axs[0].imshow(map, origin='lower')
    axs[1].imshow(map, origin='lower')

    graph_nodes_pix = [node[0] for node in graph.graph_nodes]
    colours = [("#fa8072" if node[2] else "g") for node in graph.graph_nodes]
    graph_nodes_pix = np.array(graph_nodes_pix)
    axs[0].scatter(graph_nodes_pix[:,0], graph_nodes_pix[:,1], c=colours, s=20, picker=True)
    axs[1].scatter(graph_nodes_pix[:,0], graph_nodes_pix[:,1], c=colours, s=20, picker=True)

    src, dst, _, _ = graph.graph_edges[target[1]]
    src_pt = graph.graph_nodes[src][0]
    dst_pt = graph.graph_nodes[dst][0]
    axs[0].arrow(*src_pt, *(dst_pt - src_pt), color='aqua', head_length=4, head_width=2.5)

    colourmap = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
        cmap=plt.get_cmap('cool')
        )

    probs = np.array(subgraph)[:, -1]
    predicted_edge = np.argmax(probs)
    colours = colourmap.to_rgba(probs)
    print(probs)
    # print(np.array(subgraph)[:, :2])

    for edge, colour in zip(subgraph, colours):
        src, dst, prob = edge
        start = graph.graph_nodes[int(src)][0]
        end = graph.graph_nodes[int(dst)][0]

        axs[1].arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
            linestyle=':', color=colour, head_length=4, head_width=2.5)

    pred_start_node, pred_end_node, _ = subgraph[predicted_edge]
    pred_start, _, _ = graph.graph_nodes[int(pred_start_node)]
    pred_end, _, _ = graph.graph_nodes[int(pred_end_node)]
    c = ['r', 'g']
    axs[1].scatter([pred_start[0], pred_end[0]], [pred_start[1], pred_end[1]], c=c, marker='o')
    fig.suptitle(str(image_pointer) + ' | ' + str(dist_probs[0]))

    fig_depth, axs_depth = plt.subplots(5, 3)
    fig_depth.canvas.mpl_connect('key_press_event', on_press)

    data_index = target[3]
    area_idx, file_idx = data_file_descs[data_index]
    img_idxs = data_img_idxs[data_index]

    filepath = os.path.join(
        # '/Users/joel/Research/data/ros_test/blocal_h5_smallsize_val/',
        # '/Users/joel/Research/data/ros_test/com1_basement_test/',
        '/Users/joel/Research/data/ros_test/blocal_odom_h5_test/',
        area_keys[area_idx],
        file_keys[area_idx][file_idx]
    )

    with h5py.File(filepath, 'r') as f:
        depth_ims = np.array(f['combined_depth'])
        for i in range(5):
            img_idx = img_idxs[i*2]
            imgs = depth_ims[img_idx, :, :, :]
            axs_depth[i][0].imshow(imgs[0])
            axs_depth[i][1].imshow(imgs[1])
            axs_depth[i][2].imshow(imgs[2])

    tile_figures(cols=2, rows=1, screen_rect=[0, 0, 1280, 480], tile_offsets=None)
    plt.show()