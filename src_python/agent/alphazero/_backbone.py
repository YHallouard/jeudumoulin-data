import copy
from functools import singledispatch

import torch
import torch.nn as nn
from pydantic import BaseModel


class MLPBackboneConfig(BaseModel):
    model_type: str = "mlp_backbone"
    input_dim: int = 77
    num_layers: int = 2
    hidden_dim: int = 256
    output_dim: int = 256


class MLPBackbone(nn.Module):
    def __init__(self, config: MLPBackboneConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, self.config.hidden_dim),
            nn.RMSNorm(self.config.hidden_dim),
            nn.ReLU(),
        )

        self.block = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.RMSNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.RMSNorm(self.config.hidden_dim),
        )
        self.blocks = nn.ModuleList([copy.deepcopy(self.block) for _ in range(config.num_layers)])

        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
            nn.RMSNorm(self.config.output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)

        x = self.output_proj(x)
        return x


class GraphConvLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, adjacency: torch.Tensor, use_residual: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("adjacency", adjacency)
        self.use_residual = use_residual and (in_features == out_features)

        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        x = torch.matmul(self.adjacency, x)
        x = torch.matmul(x, self.weight) + self.bias
        x = self.norm(x)
        if residual is not None:
            x = x + residual
        return torch.relu(x)


class GraphConvBackboneConfig(BaseModel):
    model_type: str = "graph_conv_backbone"
    player_embedding_dim: int = 16
    phase_embedding_dim: int = 16
    board_embedding_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 256
    graph_layer_hidden_dim: int = 32
    graph_layer_output_dim: int = 64
    num_graph_layers: int = 3
    use_attention_pooling: bool = True


class GraphConvBackbone(nn.Module):
    def __init__(self, config: GraphConvBackboneConfig) -> None:
        super().__init__()
        self.config = config

        self.player_embedding = nn.Embedding(2, config.player_embedding_dim)
        self.phase_embedding = nn.Embedding(3, config.phase_embedding_dim)
        self.board_encoder = nn.Linear(72, config.board_embedding_dim)

        adjacency_matrix: torch.Tensor = self._build_adjacency_matrix()
        self.register_buffer("adjacency", adjacency_matrix)

        self.graph_layers = nn.ModuleList()
        if not isinstance(self.adjacency, torch.Tensor):
            raise TypeError("Adjacency matrix must be a torch.Tensor")  # noqa: TRY003
        self.graph_layers.append(GraphConvLayer(3, self.config.graph_layer_hidden_dim, self.adjacency))

        for i in range(1, config.num_graph_layers):
            in_dim = self.config.graph_layer_hidden_dim
            out_dim = (
                self.config.graph_layer_output_dim
                if i == config.num_graph_layers - 1
                else self.config.graph_layer_hidden_dim
            )
            self.graph_layers.append(GraphConvLayer(in_dim, out_dim, self.adjacency, use_residual=(in_dim == out_dim)))

        self.attention_pool: nn.Sequential | None = None
        if config.use_attention_pooling:
            self.attention_pool = nn.Sequential(
                nn.Linear(self.config.graph_layer_output_dim, self.config.graph_layer_output_dim),
                nn.Tanh(),
                nn.Linear(self.config.graph_layer_output_dim, 1),
            )

        fusion_input_dim = (
            self.config.board_embedding_dim
            + self.config.graph_layer_output_dim
            + self.config.player_embedding_dim
            + self.config.phase_embedding_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
        )

    def _build_adjacency_matrix(self) -> torch.Tensor:
        # Matrice d'adjacence pour le Jeu du Moulin (24x24)
        # Chaque position connectée à ses voisines dans le graphe
        adj = torch.zeros(24, 24)

        # Connections du plateau (exemple simplifié)
        # Carré extérieur (positions 0-7)
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 0),
            # Carré moyen (positions 8-15)
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 8),
            # Carré intérieur (positions 16-23)
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 16),
            # Connections entre carrés
            (1, 9),
            (9, 17),
            (3, 11),
            (11, 19),
            (5, 13),
            (13, 21),
            (7, 15),
            (15, 23),
        ]

        for i, j in connections:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        # Normalisation (D^{-1/2} A D^{-1/2})
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        adj_normalized = adj * degree_inv_sqrt.unsqueeze(1) * degree_inv_sqrt.unsqueeze(0)

        return adj_normalized

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        player = self.player_embedding(torch.argmax(state[0:2]).long())
        phase = self.phase_embedding(torch.argmax(state[2:5]).long())
        board = self.board_encoder(state[5:])

        board_state = state[5:].reshape(24, 3)
        graph_features = board_state

        for layer in self.graph_layers:
            graph_features = layer(graph_features)

        if self.attention_pool is not None:
            attention_weights = self.attention_pool(graph_features)
            attention_weights = torch.softmax(attention_weights, dim=0)
            graph_global = torch.sum(graph_features * attention_weights, dim=0)
        else:
            graph_global = torch.max(graph_features, dim=0)[0]

        x = torch.cat([board, graph_global, player, phase])
        x = self.fusion(x)
        return x


BackboneConfig = MLPBackboneConfig | GraphConvBackboneConfig


@singledispatch
def get_backbone(config: BackboneConfig) -> nn.Module:
    raise NotImplementedError(f"Unsupported policy head config type: {type(config)}")


@get_backbone.register
def _(config: MLPBackboneConfig) -> nn.Module:
    return MLPBackbone(config)


@get_backbone.register
def _(config: GraphConvBackboneConfig) -> nn.Module:
    return GraphConvBackbone(config)
