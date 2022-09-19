import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from structures import Node, Intention

class EdgeBlock(nn.Module):
    """This module updates the edge attributes.
    The inputs are the edge, vertex, and global attributes.
    """

    def __init__(self, global_feat_size, node_feat_size, edge_feat_size, out_edge_feat_size):
        super(EdgeBlock, self).__init__()
        self.fc1 = nn.Linear(global_feat_size + 2 * node_feat_size + edge_feat_size, 256)
        self.fc1_bn = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(256, 256)
        self.fc2_bn = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(256, 256)
        self.fc3_bn = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(256, out_edge_feat_size)

    def forward(self, x):
        """Forward pass through the edge block to get updated edge attributes.
        """
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        return x


class NodeBlock(nn.Module):
    """This module updates the node attributes.
    The inputs are the global, vertex, and (updated and aggregated) edge attributes.
    """

    def __init__(self, global_feat_size, node_feat_size, edge_feat_size, out_node_feat_size):
        super(NodeBlock, self).__init__()
        self.f_n = nn.Sequential(
            nn.Linear(global_feat_size + node_feat_size + edge_feat_size, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, out_node_feat_size),
            )

    def forward(self, x):
        """Forward pass through the node block to get updated node attributes.
        """
        return self.f_n(x)


class GlobalBlock(nn.Module):
    """This module updates the global attributes.
    The inputs are the global, (updated and aggregated) vertex, and (updated and aggregated)
    edge attributes.
    """

    def __init__(self, global_feat_size, node_feat_size, edge_feat_size, out_global_feat_size):
        super(GlobalBlock, self).__init__()
        self.f_g = nn.Sequential(
            nn.Linear(global_feat_size + node_feat_size + edge_feat_size, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, out_global_feat_size),
            )

    def forward(self, x):
        """Forward pass through the global block to get updated global attributes.
        """
        return self.f_g(x)


class GNBlock(nn.Module):
    """Graph Net Block. A single graph net layer.
    The inputs are the global, vertex, and edge attributes.
    The outputs are the updated global, vertex, and edge attributes.
    """

    def __init__(self, global_feat_size, node_feat_size, edge_feat_size, out_global_feat_size,
                 out_node_feat_size, out_edge_feat_size, aggregate_method='sum'):
        """Constructor.
        Args:
            global_feat_size: Dimension of global features input to the GNBlock.
            node_feat_size: Dimension of node features input to the GNBlock.
            edge_feat_size: Dimension of edge features input to the GNBlock.
            out_global_feat_size: Size of output global features. This can be None, which saves
                    computation by skipping over the global block entirely.
            out_node_feat_size: Dimension of node features output from the GNBlock.
            out_edge_feat_size: Dimension of edge features output from the GNBlock.
            aggregate_method: Method of aggregating/pooling node and edge features ('sum' or 'avg').
        """
        super(GNBlock, self).__init__()
        self.global_feat_size = global_feat_size
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.aggregate_method = aggregate_method
        self.out_global_feat_size = out_global_feat_size
        assert (self.aggregate_method == 'sum') or (self.aggregate_method == 'avg')

        self.edge_block = EdgeBlock(self.global_feat_size, self.node_feat_size, self.edge_feat_size,
                                    out_edge_feat_size)
        self.node_block = NodeBlock(self.global_feat_size, self.node_feat_size, out_edge_feat_size,
                                    out_node_feat_size)
        if self.out_global_feat_size is not None:
            self.global_block = GlobalBlock(self.global_feat_size, out_node_feat_size,
                                            out_edge_feat_size, self.out_global_feat_size)

    def _forward_batch(subgraph_batch, image_stack):
        """Forward pass a batch. Try to do this efficiently by computing a single pass through each
        graph net block (node block, edge block, global block).
        """
        pass

    def _forward_single(subgraph_batch, image_stack):
        """Forward pass a single input instance.
        """
        pass

    def forward(self, global_features, node_features, edge_features, edge_connections,
                graph_idx_of_node, graph_idx_of_edge):
        """Forward pass through the graph net block, updating the features/attributes in-place.
        """
        device = global_features.device

        # Update edge features
        start_node_features = F.embedding(edge_connections[:, 0], node_features)
        end_node_features = F.embedding(edge_connections[:, 1], node_features)
        global_features_by_edge = F.embedding(graph_idx_of_edge, global_features)
        edge_block_input = torch.cat([global_features_by_edge, start_node_features,
                                      end_node_features, edge_features], dim=1)
        new_edge_features = self.edge_block(edge_block_input)

        # Update node features
        # Aggregate updated edge features for each node
        # Edges are directed, so only aggregate on receiver nodes
        aggregated_edge_features_per_node = torch.zeros(node_features.size(0),
                                                        new_edge_features.size(1),
                                                        device=device)
        end_node_idx = edge_connections[:, 1].view(-1, 1).expand_as(new_edge_features)
        aggregated_edge_features_per_node.scatter_add_(0, end_node_idx, new_edge_features)

        if self.aggregate_method == 'avg':
            raise NotImplementedError()

        global_features_by_node = F.embedding(graph_idx_of_node, global_features)
        node_block_input = torch.cat([global_features_by_node, node_features,
                                      aggregated_edge_features_per_node], dim=1)
        new_node_features = self.node_block(node_block_input)

        # Update global features
        if self.out_global_feat_size is None:
            new_global_features = None
        else:
            aggregated_edge_features_per_graph = torch.zeros(global_features.size(0),
                                                             new_edge_features.size(1),
                                                             device=device)
            aggregated_node_features_per_graph = torch.zeros(global_features.size(0),
                                                             new_node_features.size(1),
                                                             device=device)
            graph_idx_of_edge_exp = graph_idx_of_edge.view(-1, 1).expand_as(new_edge_features)
            graph_idx_of_node_exp = graph_idx_of_node.view(-1, 1).expand_as(new_node_features)
            aggregated_edge_features_per_graph.scatter_add_(0, graph_idx_of_edge_exp, new_edge_features)
            aggregated_node_features_per_graph.scatter_add_(0, graph_idx_of_node_exp, new_node_features)

            if self.aggregate_method == 'avg':
                raise NotImplementedError()

            global_block_input = torch.cat([global_features, aggregated_node_features_per_graph,
                                            aggregated_edge_features_per_graph], dim=1)
            new_global_features = self.global_block(global_block_input)

        return new_global_features, new_node_features, new_edge_features


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_dim):
        super(CNNBlock, self).__init__()

        # Input resolution: 320x240
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv6_bn = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(in_channels=512, out_channels=out_dim, kernel_size=(2, 3), stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=out_dim, kernel_size=(1, 3), stride=1, padding=0)

    def freeze_all_but_last_layer(self):
        print('--> Freezing the following CNNBlock parameters:')
        unfrozen_parameter_names = []
        for name, param in self.named_parameters():
            if not name.startswith('conv7'):
                print('        {}'.format(name))
                param.requires_grad = False
            else:
                unfrozen_parameter_names.append(name)
        print('--> The following parameters of CNNBlock are NOT frozen:')
        for name in unfrozen_parameter_names:
            print('        {}'.format(name))
        print('')

    def get_params_for_all_but_last_layer(self):
        names_list = []
        params_list = []
        for name, param in self.named_parameters():
            if not name.startswith('conv7'):
                names_list.append(name)
                params_list.append(param)
        return names_list, params_list

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = torch.tanh(self.conv7(x))
        x = torch.squeeze(x, dim=3)
        x = torch.squeeze(x, dim=2)
        return x

class GLN(pl.LightningModule):
    def __init__(
        self,
        num_layers,
        gn_feat_size,
        input_global_feat_size,
        input_image_stack_depth,
        lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_layers = num_layers
        self.lr = lr

        self.cnn_encoder = CNNBlock(in_channels=input_image_stack_depth, out_dim=input_global_feat_size)
        self.node_embeddings = nn.Embedding(num_embeddings=len(Node),
                                            embedding_dim=gn_feat_size)
        self.edge_embeddings = nn.Embedding(num_embeddings=len(Intention),
                                            embedding_dim=gn_feat_size)
        self.gn_layers = self.build_gn_layers(num_layers, gn_feat_size, input_global_feat_size)

    def build_gn_layers(self, num_layers, gn_feat_size, input_global_feat_size):
        layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:                
                gn_block = GNBlock(
                    global_feat_size=input_global_feat_size, node_feat_size=gn_feat_size,
                    edge_feat_size=gn_feat_size, out_global_feat_size=gn_feat_size,
                    out_node_feat_size=gn_feat_size, out_edge_feat_size=gn_feat_size
                )
            elif layer_idx == num_layers - 1:
                gn_block = GNBlock(
                    global_feat_size=gn_feat_size, node_feat_size=gn_feat_size,
                    edge_feat_size=gn_feat_size, out_global_feat_size=None,
                    out_node_feat_size=1, out_edge_feat_size=1
                )
            else:
                gn_block = GNBlock(*[gn_feat_size for i in range])

            layers.append(gn_block)
        return layers

    def forward(self, image_stack, input_graph):
        node_categories, edge_categories, edge_connections, node_idxs, edge_idxs = input_graph

        # Forward pass through CNN encoder
        initial_global_feature_batch = self.cnn_encoder(image_stack)  # batch_size x global_feat_dim

        global_features = initial_global_feature_batch
        node_features = self.node_embeddings(node_categories)
        edge_features = self.edge_embeddings(edge_categories)
        for layer in self.gn_layers:
            global_features, node_features, edge_features = layer(
                global_features,
                node_features,
                edge_features,
                edge_connections,
                node_idxs,
                edge_idxs
            )
        return global_features, node_features, edge_features

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def edge_cross_entropy(self, input, target, batch_edge_count_summed):
        batch_size = len(batch_edge_count_summed) - 1
        losses = torch.empty(batch_size, dtype=torch.float32, device=self.device)

        for idx in range(batch_size):
            start_idx = batch_edge_count_summed[idx]
            end_idx = batch_edge_count_summed[idx+1]
            class_vec = input[start_idx:end_idx]
            losses[idx] = -class_vec[target[idx]] + torch.log(torch.sum(torch.exp(class_vec)))
        
        averaged = torch.mean(losses)
        return averaged

    def edge_softmax(self, input, batch_edge_count_summed):
        batch_size = len(batch_edge_count_summed) - 1
        edge_probs = torch.empty(input.shape, dtype=torch.float32, device=self.device)

        for idx in range(batch_size):
            start_idx = batch_edge_count_summed[idx]
            end_idx = batch_edge_count_summed[idx+1]
            class_vec = input[start_idx:end_idx]

            normalizer = torch.log(torch.sum(torch.exp(class_vec)))
            edge_probs[start_idx:end_idx, :] = torch.exp(class_vec - normalizer)

        return edge_probs

    def training_step(self, batch, batch_idx):
        target, image_stack, input_graph, _, batch_edge_count_summed = batch
        batch_size = target.shape[0]
        _, _, edge_feats = self.forward(image_stack, input_graph)

        # Compute loss
        loss = self.edge_cross_entropy(edge_feats, target, batch_edge_count_summed)
        
        self.log("train_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        # self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)
        # self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        return loss

    def predict_step(self, batch, batch_idx):
        target = None
        if len(batch) == 5:
            _, image_stack, input_graph, _, batch_edge_count_summed = batch
        else:
            image_stack, input_graph, batch_edge_count_summed = batch

        _, _, edge_feats = self.forward(image_stack, input_graph)
        edge_probs = self.edge_softmax(edge_feats, batch_edge_count_summed)

        if len(batch) == 5:
            target, _, _, graph_edge_conns, _ = batch
            return target, graph_edge_conns, edge_probs, batch_edge_count_summed
        else:
            return edge_probs, batch_edge_count_summed

    def test_step(self, batch, batch_idx):
        target, image_stack, input_graph, _, batch_edge_count_summed = batch
        _, _, edge_feats = self.forward(image_stack, input_graph)

        # Compute loss
        loss = self.edge_cross_entropy(edge_feats, target, batch_edge_count_summed)
        return loss        

    def validation_step(self, batch, batch_idx):
        target, image_stack, input_graph, _, batch_edge_count_summed = batch
        batch_size = target.shape[0]
        _, _, edge_feats = self.forward(image_stack, input_graph)

        # Compute loss
        loss = self.edge_cross_entropy(edge_feats, target, batch_edge_count_summed)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        return loss
