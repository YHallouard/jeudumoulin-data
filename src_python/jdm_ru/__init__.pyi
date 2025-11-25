"""Type stubs for the jdm_ru Rust module."""

from typing import Protocol

class PyBoard:
    """
    Represents the game board state.

    The board is a 24-position grid where players place and move pieces.
    """

    def __init__(self) -> None:
        """Create a new board in the initial state."""
        ...

    def legal_moves(self) -> list[PyMove]:
        """
        Get all legal moves for the current player.

        Returns:
            List of legal PyMove objects
        """
        ...

    def apply_move(self, move: PyMove) -> PyBoard:
        """
        Apply a move and return the resulting board state.

        Args:
            move: The move to apply

        Returns:
            New PyBoard with the move applied
        """
        ...

    def is_terminal(self) -> bool:
        """
        Check if the game is over.

        Returns:
            True if the game has ended
        """
        ...

    def winner(self) -> int | None:
        """
        Get the winner of the game if it's over.

        Returns:
            1 for White, -1 for Black, None if no winner yet
        """
        ...

    def to_embed(self) -> list[float]:
        """
        Convert board state to a flat embedding vector.

        Returns a 77-element vector:
        - [0-1]: One-hot encoded current player (White, Black)
        - [2-4]: One-hot encoded phase (Placing, Moving, Flying)
        - [5-76]: One-hot encoded board squares (24 positions x 3 states each)

        Returns:
            List of 77 floats representing the board state
        """
        ...

    def current_player(self) -> int:
        """
        Get the current player.

        Returns:
            1 for White, -1 for Black
        """
        ...

    def print_board(self) -> None:
        """Print a visual representation of the board to stdout."""
        ...

class PyMove:
    """
    Represents a move in the game.

    A move can be:
    - A placement (to_position only)
    - A piece movement (from_position and to_position)
    - Include a capture (removed_position)
    """

    def __init__(self, from_position: int | None, to_position: int, removed_position: int | None) -> None:
        """
        Create a new move.

        Args:
            from_position: Source position (None for placement moves)
            to_position: Destination position (0-23)
            removed_position: Position to remove opponent piece (None if no capture)
        """
        ...

    @staticmethod
    def placement(position: int) -> PyMove:
        """
        Create a placement move.

        Args:
            position: Position to place the piece (0-23)

        Returns:
            PyMove representing a placement
        """
        ...

    @staticmethod
    def move_piece(from_position: int, to_position: int) -> PyMove:
        """
        Create a movement move.

        Args:
            from_position: Source position (0-23)
            to_position: Destination position (0-23)

        Returns:
            PyMove representing a piece movement
        """
        ...

    def with_removal(self, removed_position: int) -> PyMove:
        """
        Create a copy of this move with a piece removal.

        Args:
            removed_position: Position to remove opponent piece (0-23)

        Returns:
            New PyMove with removal
        """
        ...

    def to_embed(self) -> list[float]:
        """
        Convert move to a flat embedding vector.

        Returns a 72-element vector:
        - [0-23]: One-hot encoded from_position
        - [24-47]: One-hot encoded to_position
        - [48-71]: One-hot encoded removed_position

        Returns:
            List of 72 floats representing the move
        """
        ...

    def to_indices(self) -> list[int]:
        """
        Convert move to compact indices representation.
        Returns [from, to, remove] where each is 0-24 (24 = None).
        """
        ...

    def from_position(self) -> int | None:
        """Get the source position (None for placement moves)."""
        ...

    def to_position(self) -> int:
        """Get the destination position."""
        ...

    def removed_position(self) -> int | None:
        """Get the removed position (None if no capture)."""
        ...

    def __repr__(self) -> str:
        """Return string representation of the move."""
        ...

    def __hash__(self) -> int:
        """Return hash of the move for use as dictionary key."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check if two moves are equal."""
        ...

    def __ne__(self, other: object) -> bool:
        """Check if two moves are not equal."""
        ...

class PyNode:
    """
    MCTS search tree node.
    """

    @property
    def children(self) -> dict[PyMove, PyNode]:
        """
        Get the children of this node.

        Returns:
            Dictionary of PyMove to PyNode
        """
        ...

    @property
    def prior(self) -> float:
        """
        Get the prior of this node

        Returns:
            Prior value of this Node
        """
        ...

    @property
    def value(self) -> float:
        """
        Get the average value of this node.

        Returns:
            Average value from all visits
        """
        ...

    @property
    def visit_count(self) -> int:
        """
        Get the number of times this node has been visited.

        Returns:
            Visit count
        """
        ...

    def select_action(self, temperature: float) -> PyMove:
        """
        Select an action based on visit counts and temperature.

        Args:
            temperature: Controls randomness (0 = greedy, higher = more random)

        Returns:
            Selected move
        """
        ...

class PyMCTS:
    """
    Monte Carlo Tree Search implementation.
    """

    def __init__(self, num_simulations: int, show_progress: bool = False) -> None:
        """
        Create a new MCTS instance.

        Args:
            num_simulations: Number of simulations to run per search
            show_progress: Whether to show progress bar
        """
        ...

    def run(
        self,
        agent: AgentProtocol,
        board: PyBoard,
        depth: int,
        root: PyNode | None = None,
    ) -> PyNode:
        """
        Run MCTS search from the given board state.

        Args:
            agent: Agent to use for evaluation
            board: Current board state
            depth: Current depth in the game tree
            root: Optional root node to continue search from

        Returns:
            Root node after search
        """
        ...

class AgentProtocol(Protocol):
    """
    Protocol defining the interface for agents used in self-play training.
    """

    def predict(
        self, state_embedding: list[float], legal_moves: list[list[int | None]]
    ) -> tuple[dict[int, float], float]:
        """
        Predict policy and value for a given state.

        Args:
            state_embedding: Board state as 77 floats
            legal_moves: Legal moves as list of [from, to, removed] positions

        Returns:
            Tuple of (policy_dict, value) where:
            - policy_dict: Maps move index to probability
            - value: Value estimate for the state
        """
        ...

def generate_train_examples(
    agent: AgentProtocol,
    num_simulations: int,
    num_episodes: int,
    max_episode_steps: int,
    temperature: float,
) -> tuple[list[list[float]], list[list[list[int | None]]], list[list[float]], list[float]]:
    """
    Generate training examples through self-play.

    Args:
        agent: Agent implementing predict(state_embedding, legal_moves) method
        num_simulations: Number of MCTS simulations per move
        num_episodes: Number of episodes to generate
        max_episode_steps: Maximum steps per episode
        temperature: Temperature for action selection (higher = more random)

    Returns:
        Tuple of (state_embeddings, legal_moves, policy_labels, value_labels) where:
        - state_embeddings: Board embeddings, each is 77 floats
        - legal_moves: Legal moves for each state, each move is [from, to, removed]
        - policy_labels: Policy probability distributions over legal moves
        - value_labels: Value labels (-1 for loss, 0 for draw, +1 for win)
    """
    ...
