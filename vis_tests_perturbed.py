import os
import sys
import h5py
import json
import argparse
import numpy as np

from structures import BehaviourGraph
from map_utils import mapPoint2Pixel

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def visualise_file(
    file_results, 
    file_traj, 
    dirname,
    filename,
    graph, 
    map, 
    map_bounds, 
    map_res, 
    img_stack_size
    ):
    targets = np.array(file_results['targets'])
    results = np.array(file_results['subgraph_edge_output'])

    colourmap = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
        cmap=plt.get_cmap('cool')
        )
    it = enumerate(zip(targets, results))

    segment_count = int(np.array(file_traj['segment_count']))
    segments = [np.array(file_traj[str(idx) + '/points'])[:, :3] for idx in range(segment_count)]
    segment_cumlens = np.cumsum(np.array([len(segment) for segment in segments]))
    segment_lower_bounds = np.insert(segment_cumlens, 0, 0)[:-1]
    point_history = np.array([point for segment in segments for point in segment])

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(map, origin='lower')
    axs[1].imshow(map, origin='lower')

    def on_click(event):
        if event.key == "enter":
            try:
                idx, (target, result) = next(it)
            except StopIteration:
                plt.close('all')
            recent_img_point_idx, recent_img_segment_idx, graph_edge_idx, gt_edge_idx = target
            result = np.array(result)
            predicted_edge = np.argmax(result[:, -1])
            colours = colourmap.to_rgba(result[:, -1])

            # Redraw ground truth
            axs[0].cla()
            axs[0].imshow(map, origin='lower')

            for segment_idx in range(segment_count):
                edge_idx = int(np.array(file_traj[str(segment_idx) + '/edge_idx']))
                src, dst, _, _ = graph.graph_edges[edge_idx]
                src_pt, _, _ = graph.graph_nodes[src]
                dst_pt, _, _ = graph.graph_nodes[dst]
                c = 'aqua' if edge_idx == graph_edge_idx else 'k'
                axs[0].arrow(*src_pt, *(dst_pt - src_pt), color=c, head_length=4, head_width=2.5)

            nearest_img_point_idx = recent_img_point_idx + segment_lower_bounds[recent_img_segment_idx]
            oldest_img_point_idx = nearest_img_point_idx - img_stack_size
            img_stack_endpoints = point_history[[nearest_img_point_idx, oldest_img_point_idx]]
            img_stack_coords = np.array([mapPoint2Pixel(pt, map_bounds, map_res) for pt in img_stack_endpoints])
            axs[0].scatter(img_stack_coords[:, 0], img_stack_coords[:, 1], marker='x')

            # Redraw GLN output
            axs[1].cla()
            axs[1].imshow(map, origin='lower')

            for colour, (start_node, end_node, prob)  in zip(colours, result):
                start, _, _ = graph.graph_nodes[int(start_node)]
                end, _, _ = graph.graph_nodes[int(end_node)]
                axs[1].arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
                    color=colour, head_length=4, head_width=2.5)

            pred_start_node, pred_end_node, _ = result[predicted_edge]
            pred_start, _, _ = graph.graph_nodes[int(pred_start_node)]
            pred_end, _, _ = graph.graph_nodes[int(pred_end_node)]
            c = ['r', 'g']
            axs[1].scatter([pred_start[0], pred_end[0]], [pred_start[1], pred_end[1]], c=c, marker='o')

            fig.suptitle(dirname + " " + filename + ": " + str(idx))
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_click)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="Path to test results",
        default="/Users/joel/Research/behaviour_mapping/blocal/results_area3_perturb.json")
    parser.add_argument("--test_dir", type=str, help="Path to test dataset",
        default="/Users/joel/Research/data/blocal_val_area3_perturb")
    parser.add_argument("--depth_stack_size", type=int, help="Size of the depth image stack",
        default=20)
    args = parser.parse_args()

    data_map = None
    with open(args.results_dir, 'r') as f:
        data_map = json.load(f)
    
    if data_map is None:
        print("No data!")
        sys.exit(0)

    for dirname, dirmap in data_map.items():
        for graphname, graph_results in dirmap.items():
            graphname = os.path.basename(graphname)
            graph_path = os.path.join(args.test_dir, dirname, graphname)
            graph = BehaviourGraph()
            graph.loadGraph(graph_path)

            map_path = os.path.join(args.test_dir, dirname, "map.npz")
            tmp = np.load(map_path)
            map = tmp['map']
            map_bounds = tmp['bounds']
            map_res = tmp['res']

            print("=====")
            print(dirname, ": ", graphname)
            print("=====")
            print(graph_results['description'])

            if graphname != 'behaviour_graph.json':
                continue

            traj_file = [f for f in os.listdir(os.path.join(args.test_dir, dirname)) if f.endswith('.h5')]
            assert(len(traj_file) == 1)
            traj_file = os.path.join(args.test_dir, dirname, traj_file[0])
            print(args.test_dir, dirname, traj_file)
            with h5py.File(traj_file, 'r') as f:
                visualise_file(
                    graph_results, f, dirname, graphname,
                    graph, map, map_bounds, map_res, 
                    args.depth_stack_size
                    )