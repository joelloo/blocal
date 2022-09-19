import os
import json
import heapq
import numpy as np
import bisect
from enum import IntEnum
from copy import deepcopy

class Node(IntEnum):
    CHANGEPOINT = 0
    PLACE = 1

class Intention(IntEnum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2

class BehaviourGraph:
    def __init__(self, scene_dir=""):
        self.scene_dir = scene_dir
        self.graph_nodes = []
        self.graph_edges = []

        self.vertices = []
        self.graph_edges_map = {}
        self.cumulative_edge_dists = []

        self.rng = None

    # I/O
    def writeGraph(self):
        basename = os.path.basename(self.scene_dir)
        dirname = os.path.dirname(self.scene_dir)
        scenename = basename.split(".")[0]
        filename = scenename + "_graph.json"
        outfile = os.path.join(dirname, filename)

        flattened_graph_nodes = [(*pix, *pos) for pix, pos, _ in self.graph_nodes]
        pos_graph_nodes = list(map(lambda tup: tuple(map(float, tup)), flattened_graph_nodes))
        labels_graph_nodes = [place for _, _, place in self.graph_nodes]
        graph_edges = [tuple(map(int, tup)) for tup in self.graph_edges]
        data = {'scene_name': scenename,
                'scene_dir': self.scene_dir,
                'graph_node_pos': pos_graph_nodes,
                'graph_node_labels': labels_graph_nodes,
                'graph_edges': graph_edges }

        print("Writing data to ", outfile)
        with open(outfile, 'w') as f:
            json.dump(data, f)

    def loadGraph(self, graph_dir):
        print("Loading data from ", graph_dir)
        with open(graph_dir, 'r') as f:
            data = json.load(f)

            print("Setting scene as ", data['scene_name'])
            pos = data['graph_node_pos']
            labels = data['graph_node_labels']
            self.scene_dir = data['scene_dir']
            self.graph_nodes = [(np.array([p[0], p[1]]), [p[2], p[3], p[4]], l) for p, l in zip(pos, labels)]
            self.graph_edges = [(e[0], e[1], Intention(e[2]), e[3]) for e in data['graph_edges']]

    # Sampling and graph search
    def initialise(self, random_seed=0):
        self.vertices = [{
            'coord': np.array(pos),
            'place_node': place,
            'out_edges': []
            } for _, pos, place in self.graph_nodes]

        edge_dists = [
            np.linalg.norm(self.vertices[dst]['coord'] - self.vertices[src]['coord'])
            for src, dst, _, _ in self.graph_edges
        ]
        self.cumulative_edge_dists = np.cumsum(np.array(edge_dists))

        for (edge_src, edge_dst, edge_int, _), dist in zip(self.graph_edges, edge_dists):
            self.vertices[edge_src]['out_edges'].append(
                (edge_dst, dist, edge_int)
            )

        for edge_idx, (edge_src, edge_dst, edge_int, _) in enumerate(self.graph_edges):
            if (edge_src, edge_dst) in self.graph_edges_map.keys():
                print(edge_src, edge_dst, edge_idx)
                raise Exception("Duplicate edges in graph!")
            self.graph_edges_map[(edge_src, edge_dst)] = (edge_int, edge_idx)

        # Sanity check correctness of graph -- make sure all edges are reversible,
        # i.e. for any edge (src, dst), there also exists the corresponding edge
        # (dst, src)
        for edge_idx, (edge_src, edge_dst, _, _) in enumerate(self.graph_edges):
            if (edge_dst, edge_src) not in self.graph_edges_map.keys():
                print(edge_src, edge_dst, edge_idx)
                raise Exception("Edge not reversible!")

        self.rng = np.random.default_rng(random_seed)


    def dijkstra(self, src, dst):
        dists = [0 if idx == src else np.inf for idx in range(len(self.vertices))]
        prevs = [None for _ in range(len(self.vertices))]
        pq = [(dist, idx) for idx, dist in enumerate(dists)]
        heapq.heapify(pq)

        while len(pq) > 0:
            min_dist, min_idx = pq[0]
            if min_idx == dst:
                pointer = dst
                path = []
                coords = []
                labels = []
                while pointer is not None:
                    path.append(pointer)
                    coords.append(self.vertices[pointer]['coord'])
                    tup = prevs[pointer]
                    if tup is not None:
                        pointer, intention = tup
                        labels.append(intention)
                    else:
                        pointer = None

                path.reverse()
                coords.reverse()
                labels.reverse()

                return path, np.array(coords), labels
                    
            heapq.heappop(pq)

            for nidx, nlen, intention in self.vertices[min_idx]['out_edges']:
                alt = dists[min_idx] + nlen
                if alt < dists[nidx]:
                    dists[nidx] = alt
                    prevs[nidx] = (min_idx, intention)
                    pidx = next(i for i, elem in enumerate(pq) if elem[1] == nidx)
                    pq[pidx] = (alt, nidx)

            heapq.heapify(pq)

        return None

    def sampleEdge(self):
        if len(self.cumulative_edge_dists) < 1:
            return None

        # sampled_dist = np.random.random() * self.cumulative_edge_dists[-1]
        sampled_dist = self.rng.random() * self.cumulative_edge_dists[-1]
        upper_idx = bisect.bisect_right(self.cumulative_edge_dists, sampled_dist)
        sampled_edge_dist_norm = (
            sampled_dist / self.cumulative_edge_dists[upper_idx] if upper_idx == 0 else
            (sampled_dist - self.cumulative_edge_dists[upper_idx - 1]) / (self.cumulative_edge_dists[upper_idx] - self.cumulative_edge_dists[upper_idx - 1])
        )

        edge_src, edge_dst, _, _ = self.graph_edges[upper_idx]
        src_point = self.vertices[edge_src]['coord']
        dst_point = self.vertices[edge_dst]['coord']
        sampled_point = src_point + (dst_point - src_point) * sampled_edge_dist_norm

        return sampled_point, edge_src, edge_dst