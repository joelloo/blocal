import os
import sys
import h5py
import numpy as np
import argparse
from functools import reduce

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import habitat_sim

from structures import Intention, BehaviourGraph
from map_utils import (
    map2MeshFrame,
    mapPoint2Pixel, 
    mesh2MapFrame, 
    convertMapFrame, 
    mesh2MapFrameQuaternion
)

class MapNavigator:
    def __init__(self, 
        data_dir,
        visualise=False,
        verbose=False, 
        write_dir="", 
        save_traj_ims=False,
        random_seed=0,
        min_path_length=5.0
    ):
        self.visualise = visualise
        self.verbose = verbose
        self.write_output = not (write_dir == "")
        self.save_traj_ims = save_traj_ims
        self.output_dir = write_dir
        self.min_path_length = min_path_length

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        graph_dir = os.path.join(data_dir, 'behaviour_graph.json')
        self.graph = BehaviourGraph()
        self.graph.loadGraph(graph_dir)
        self.graph.initialise(random_seed)

        self.setupSim(os.path.join(data_dir, self.graph.scene_dir))

        map_dir = os.path.join(data_dir, 'map.npz')
        tmp = np.load(map_dir)
        self.res = tmp['res']
        self.bounds = tmp['bounds']
        self.map = tmp['map']
        self.map_dims = self.map.shape
        
        print("Loaded map...")
        print("Resolution: ", self.res)
        print("Bounds: ", self.bounds)
        print("Shape of map: ", self.map_dims)


    ##################################################
    ### Helper functions for simulator interaction ###
    ##################################################
    # These functions provide an interface to the simulator
    # and handle transformations between the simulator mesh
    # frames and our own map frame.

    def convertMap2Pixel(self, point):
        return mapPoint2Pixel(point, self.bounds, self.res)


    def getFeasibleGeodesicPath(self, path):
        assert(len(path) >= 3)

        gp = habitat_sim.nav.GreedyGeodesicFollower(
            self.sim.pathfinder, self.sim.get_agent(0), goal_radius=0.25
        )

        path_data = []
        obs_data = []

        q = habitat_sim.utils.common.quat_from_two_vectors(
            habitat_sim.geo.FRONT,
            map2MeshFrame(path[1] - path[0])
        )
        agent_state = self.sim.get_agent(0).get_state()
        agent_state.position = map2MeshFrame(path[0])
        agent_state.rotation = q
        self.sim.get_agent(0).set_state(agent_state)

        if self.verbose:
            angle, axis = habitat_sim.utils.common.quat_to_angle_axis(q)
            if axis[1] < 0:
                print("Rotate clockwise by ", angle / np.pi * 180.0)
            else:
                print("Rotate counterclockwise by ", angle / np.pi * 180.0)

        for idx in range(1, len(path)):
            actions = gp.find_path(map2MeshFrame(path[idx]))
            if self.verbose:
                print(">>> ", actions)

            # Roll out the actions to get the geodesic path
            agent_rotation = self.sim.get_agent(0).get_state().rotation
            geodesics = [np.hstack(
                (path[idx-1], habitat_sim.utils.common.quat_to_coeffs(agent_rotation))
            )]
            observations = [self.sim.get_sensor_observations()['mid_depth']]
            for action_idx in range(len(actions) - 1):
                obs = self.sim.step(actions[action_idx])
                state = self.sim.get_agent(0).state
                position = np.array(mesh2MapFrame(state.position))
                rotation = mesh2MapFrameQuaternion(state.rotation)
                pose = np.hstack((position, habitat_sim.utils.common.quat_to_coeffs(rotation)))
                geodesics.append(pose)
                observations.append(obs['mid_depth'])

            observations = np.array(observations)
            observations = np.expand_dims(observations, axis=0) if len(observations.shape) == 2 else observations
            path_data.append(geodesics)
            obs_data.append(observations)

        obs_data = np.concatenate(obs_data, axis=0)
        return path_data, obs_data


    def isNavigable(self, point):
        return self.sim.pathfinder.is_navigable(map2MeshFrame(point))

    #################################################
    ### Utilities to sample and plot trajectories ###
    #################################################
    # These functions allow trajectories to be sampled
    # from the traversability map generated from the NavMesh,
    # and guided by the behaviour map.

    def sampleTrajectory(self, filename):
        path = None
        indices = None
        labels = None
        interp_path = None
        observations = None
        iter = 0

        while path is None or interp_path is None:
            print("Sampling start, goal points and trying to find a connecting path: ", iter + 1)
            start_coord, start_src, start_dst = self.graph.sampleEdge()
            end_coord, end_src, end_dst = self.graph.sampleEdge()

            if ((start_src, start_dst) == (end_src, end_dst) 
                or (start_src, start_dst) == (end_dst, end_src)
                or not (self.isNavigable(start_coord) 
                        and self.isNavigable(end_coord))
            ):
                iter += 1
                continue

            if self.verbose:
                print("===")
                print(start_src, start_dst, start_coord)
                print(end_src, end_dst, end_coord)
                print("===")

            indices, path, labels = self.graph.dijkstra(start_dst, end_src)
            print(">>>>> ", len(indices), len(path), len(labels))
            print(start_src, start_dst, end_src, end_dst)
            print(indices)

            if (len(path) > 1
                and start_src == indices[1]
                and start_dst == indices[0]):
                indices = indices[1:]
                path = path[1:]
                labels = labels[1:]
                start_src, start_dst = start_dst, start_src

            if (len(path) > 1
                and end_src == indices[-1]
                and end_dst == indices[-2]):
                indices = indices[:-1]
                path = path[:-1]
                labels = labels[:-1]
                end_src, end_dst = end_dst, end_src

            start_int, _ = self.graph.graph_edges_map[(start_src, start_dst)]
            end_int, _ = self.graph.graph_edges_map[(end_src, end_dst)]

            indices = [start_src, *indices, end_dst]
            path = [start_coord, *path, end_coord]
            labels = [start_int, *labels, end_int]
            print(">>>>>* ", len(indices), len(path), len(labels))
            print(indices)

            # Apply some heuristic checks to make sure the path
            # generated looks "intuitive". These are just heuristics
            # that are applied because we do not explicitly model
            # behaviour transitions at each changepoint in the current
            # data structure. This heuristic requires that each place node 
            # is not allowed to be a behaviour changepoint, and can only
            # be a start or end node.

            intermediate_nodes = indices[1:-1]
            if reduce(
                lambda x, y: x or y,
                map(lambda i: self.graph.vertices[i]['place_node'], intermediate_nodes)
            ):
                print("Invalid path (non-start/end place node). Resampling.")
                interp_path = None
                observations = None
                iter += 1
                continue

            path_array = np.array(path)
            dirs = path_array[1:, :] - path_array[:-1, :]
            dirs = dirs[:, [0, 2]]
            dirs = dirs / np.linalg.norm(dirs, axis=1).reshape(-1, 1)

            # The first two conditions make sure that if we take a right/left
            # into a particular changepoint, we don't take another right/left
            # out of that changepoint in a different heading direction. This arises
            # because we have not handled heading yet. Given 2 direction vectors of
            # consecutive line segments d1, d2, we enforce this condition by:
            # d1 x d2 < 0 to prevent LEFT->LEFT with a different heading for the 
            # second LEFT, and d1 x d2 > 0 to prevent RIGHT->RIGHT with a different
            # heading for the second RIGHT. These conditions hold for a right-handed
            # axis convention, but we swap them around because our convention is somehow
            # left-handed. The last condition prevents FORWARD->FORWARD transitions where
            # the directions of the FORWARD transitions have too much angular difference.
            # Unfortunately this removes the ability to follow very twisty corridors,
            # but allows us to prevent transitions that avoid using turning behaviours.
            # This also arises due to the lack of properly-handled heading at the nodes.
            if (reduce(
                lambda x, y: x or y,
                [((labels[i-1] == Intention.LEFT and labels[i] == Intention.LEFT)
                and np.cross(dirs[i-1], dirs[i]) < 0) for i in range(1, len(dirs))]
            )
            or reduce(
                lambda x, y: x or y,
                [((labels[i-1] == Intention.RIGHT and labels[i] == Intention.RIGHT)
                and np.cross(dirs[i-1], dirs[i]) > 0) for i in range(1, len(dirs))]
            )
            or reduce(
                lambda x, y: x or y,
                [((labels[i-1] == Intention.FORWARD or labels[i] == Intention.FORWARD)
                and np.dot(dirs[i-1], dirs[i]) < 0) for i in range(1, len(dirs))]
            )):
                print("Invalid path (geometry). Resampling.")
                interp_path = None
                observations = None
                iter += 1 
                continue

            # Try to extract a geodesic path by planning and rolling out in simulation
            try:
                interp_path, observations = self.getFeasibleGeodesicPath(path)
                
                points = np.concatenate(
                    list(map(lambda t: np.array(t)[:, 0:3], interp_path)), axis=0
                )
                dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
                path_length = np.sum(dists)

                print("Path length: ", path_length)
                if path_length < self.min_path_length:
                    interp_path = None
                    observations = None
                
            except habitat_sim.nav.greedy_geodesic_follower.errors.GreedyFollowerError:
                interp_path = None
                observations = None

            iter += 1

        print(indices)
        print(len(indices))
        print(len(path))
        print(len(interp_path))
        edge_indices = [self.graph.graph_edges_map[(src, dst)][1] for src, dst in zip(indices[:-1], indices[1:])]

        print(">>> Found valid path.")
        if self.visualise and self.save_traj_ims:
            self.plot(path, interp_path, labels, True, filename)
        elif self.visualise:
            self.plot(path, interp_path, labels, True, "")
        elif self.save_traj_ims:
            self.plot(path, interp_path, labels, False, filename)
            
        if self.write_output:
            self.write(interp_path, edge_indices, observations, labels, filename)


    def plot(self, path, interp_path, labels, view, filename):
        start = self.convertMap2Pixel(path[0])
        end = self.convertMap2Pixel(path[-1])
        path = np.array(list(map(self.convertMap2Pixel, path)))
        interp_path = [np.array(list(map(self.convertMap2Pixel, subpath))) for subpath in interp_path]

        plt.clf()
        plt.imshow(self.map, origin='lower')
        print(list(enumerate(map(len, interp_path))))
        print(list(enumerate(labels)))
        for label, subpath in zip(labels, interp_path):
            colour = '0.4' if label == Intention.FORWARD else ('b' if label == Intention.LEFT else 'r')
            plt.plot(subpath[:, 0], subpath[:, 1], c=colour)
        plt.scatter(path[1:-1, 0], path[1:-1, 1], c='dimgrey', marker='*')
        plt.scatter(start[0], start[1], c='orangered', marker='X')
        plt.scatter(end[0], end[1], c='seagreen', marker='X')
        
        if filename:
            plt.savefig(os.path.join(self.output_dir, filename + '.png'))
        if view:
            plt.show()


    def write(self, interp_path, edge_indices, observations, labels, filename):
        hf = h5py.File(os.path.join(self.output_dir, filename + '.h5'), 'w')

        print("Writing path")
        cumlen = np.cumsum(np.array(list(map(len, interp_path))))
        hf.create_dataset('segment_count', data=len(interp_path))
        hf.create_dataset('segment_cumlen', data=cumlen)

        # for idx, (points, depth_ims, label) in enumerate(zip(interp_path, observations, labels)):
        for idx, (points, edge_idx, label) in enumerate(zip(interp_path, edge_indices, labels)):
            hf.create_dataset(str(idx) + '/points', data=np.array(points))
            hf.create_dataset(str(idx) + '/label', data=int(label))
            hf.create_dataset(str(idx) + '/edge_idx', data=int(edge_idx))
            # hf.create_dataset(str(idx) + '/depth', data=np.array(depth_ims), compression="gzip")
        hf.create_dataset('combined_depth', data=observations, compression="gzip")
        hf.close()

        print("Writing images")
        saved_ims = np.concatenate([
            (ims if len(ims.shape) == 3 else np.expand_dims(ims, axis=0))
            for ims in observations
        ])
        print(saved_ims.shape)
        # np.savez(os.path.join(self.output_dir, 'test_ims.npz'), saved_ims=saved_ims)


    #########################################
    ### Setup functions for the simulator ###
    #########################################
    # Handles the boilerplate needed to initialise
    # the simulator.

    def setupSim(self, scene_dir):
        config = habitat_sim.SimulatorConfiguration()
        config.scene_id = scene_dir

        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = self.setupSensors()
        sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent_config]))
        
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=False):
            print("Failed to recompute navmesh")
            sys.exit(0)

        if not sim.pathfinder.is_loaded:
            print("Sim pathfinder not loaded!")
            sys.exit(0)

        self.sim = sim


    def setupSensors(self):
        mid_depth_sensor = habitat_sim.bindings.CameraSensorSpec()
        mid_depth_sensor.uuid = "mid_depth"
        mid_depth_sensor.resolution = [192, 320]
        mid_depth_sensor.position = 1.5 * habitat_sim.geo.UP
        mid_depth_sensor.orientation = [0., 0., 0.]
        mid_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        mid_depth_sensor.hfov = 150

        return [mid_depth_sensor]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to scene file",
                        default="/Users/joel/Research/data/area6")
                        # default="/data/home/joel/datasets/2d3ds/area3")
                        # default="/home/j/joell/datasets/bmapping_data/area3")
    parser.add_argument("--output_dir", type=str, help="Path to output directory",
                        default="/Users/joel/Research/data/area6/test")
                        # default="/data/home/joel/datasets/blocal_data/area3")
                        # default="/home/j/joell/datasets/blocal_data/area3")
    parser.add_argument("--num_samples", type=int, help="Number of trajectories to sample",
                        default=10)
    parser.add_argument("--random_seed", type=int, help="Change the random seed used for sampling",
                        default=42)
    parser.add_argument("--min_path_length", type=float, help="Minimum path length to use",
                        default=6.0)                    
    args = parser.parse_args()

    nav = MapNavigator(
        args.data_dir,
        visualise=False,
        write_dir=args.output_dir,
        save_traj_ims=True,
        random_seed=args.random_seed,
        min_path_length=args.min_path_length
    )

    # print(nav.sim.get_agent(0).agent_config.action_space)

    try:
        iter = 0    
        while iter < args.num_samples:
            nav.sampleTrajectory('t' + str(iter))
            iter += 1
            print("Trajectories sampled: ", iter)
    except KeyboardInterrupt:
        pass