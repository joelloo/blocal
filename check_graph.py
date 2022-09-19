from structures import BehaviourGraph

if __name__ == "__main__":
    # graph_dir = "/Users/joel/Research/data/area5/area5_rotated_graph.json"
    graph_dir = "/Users/joel/Research/behaviour_mapping/bmapping/area1_graph.json"
    graph = BehaviourGraph()
    graph.loadGraph(graph_dir)
    graph.initialise(0)
    print("Check passed!")