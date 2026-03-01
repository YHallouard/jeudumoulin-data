"""
Reward Calculator for DQN training in the Mill game.

This module provides reward shaping functionality based on the Rust implementation
from the original jeudumoulin project. It calculates rewards for game states and moves
to guide the DQN agent's learning process.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jdm_ru


class RewardCalculator:
    """
    Calculates rewards for DQN training based on game state transitions.

    The reward function combines multiple components:
    - Game end rewards (win/loss): ±10.0
    - Moulin formation: +1.0 (agent) / -2.0 (opponent)
    - Piece removal: +2.0 (agent removes) / -2.0 (opponent removes)
    - Protected pieces: ratio of pieces in moulins (Moving phase only)
    - Blocked pieces: ratio of opponent pieces without legal moves (Moving phase)

    All rewards are normalized based on the current game phase.
    """

    def __init__(self):
        """Initialize the reward calculator."""
        pass

    def calculate_reward(
        self,
        board: "jdm_ru.PyBoard",
        prev_board: "jdm_ru.PyBoard",
        move: "jdm_ru.PyMove",
        agent_player: int,
    ) -> float:
        """
        Calculate the total reward for a state transition.

        Args:
            board: Current board state (after move)
            prev_board: Previous board state (before move)
            move: The move that was applied
            agent_player: The agent's player ID (1 for White, -1 for Black)

        Returns:
            Normalized reward value (typically in range [-5, 5])
        """
        reward = 0.0

        # Game end reward (±10.0)
        reward += self._calculate_game_end_reward(board, agent_player)

        # Piece removal reward (±2.0)
        reward += self._calculate_removal_reward(prev_board, move, agent_player)

        # Moulin formation reward (+1.0 / -2.0)
        reward += self._calculate_moulin_formation_reward(prev_board, move, agent_player)

        # Protected pieces reward (Moving phase only)
        reward += self._calculate_protected_piece_reward(board, agent_player)

        # Blocked pieces reward (Moving phase only)
        reward += self._calculate_blocked_piece_reward(board, agent_player)

        # Normalize the reward based on game phase
        reward = self._normalize_reward(reward, prev_board)

        return reward

    def _calculate_game_end_reward(
        self,
        board: "jdm_ru.PyBoard",
        agent_player: int,
    ) -> float:
        """
        Calculate reward for winning or losing the game.

        Args:
            board: Current board state
            agent_player: The agent's player ID

        Returns:
            +10.0 for win, -10.0 for loss, 0.0 for ongoing game
        """
        winner = board.winner()
        if winner is None:
            return 0.0
        return 10.0 if winner == agent_player else -10.0

    def _calculate_removal_reward(
        self,
        prev_board: "jdm_ru.PyBoard",
        move: "jdm_ru.PyMove",
        agent_player: int,
    ) -> float:
        """
        Calculate reward for removing opponent pieces.

        Args:
            prev_board: Previous board state
            move: The move that was applied
            agent_player: The agent's player ID

        Returns:
            +2.0 if agent removes piece, -2.0 if opponent removes piece, 0.0 otherwise
        """
        if move.removed_position() is None:
            return 0.0

        current_player = prev_board.current_player()
        return 2.0 if current_player == agent_player else -2.0

    def _calculate_moulin_formation_reward(
        self,
        prev_board: "jdm_ru.PyBoard",
        move: "jdm_ru.PyMove",
        agent_player: int,
    ) -> float:
        """
        Calculate reward for forming moulins (mills).

        Args:
            prev_board: Previous board state
            move: The move that was applied
            agent_player: The agent's player ID

        Returns:
            +1.0 if agent forms moulin, -2.0 if opponent forms moulin, 0.0 otherwise
        """
        current_player = prev_board.current_player()

        # Check if the move forms a moulin
        # We need to check this on prev_board with the move
        if current_player == agent_player:
            if self._forms_moulin(prev_board, move, agent_player):
                return 1.0
        else:  # opponent's turn
            opponent = -agent_player
            if self._forms_moulin(prev_board, move, opponent):
                return -2.0

        return 0.0

    def _calculate_protected_piece_reward(
        self,
        board: "jdm_ru.PyBoard",
        agent_player: int,
    ) -> float:
        """
        Calculate reward based on the ratio of agent's pieces in moulins.
        Only applies during the Moving phase.

        Args:
            board: Current board state
            agent_player: The agent's player ID

        Returns:
            Ratio of protected pieces (0.0 to 1.0) in Moving phase, 0.0 otherwise
        """
        # Get phase from board state embedding
        phase = self._get_phase_from_board(board)

        if phase != 1:  # 1 = Moving phase
            return 0.0

        # Count agent pieces and how many are in moulins
        agent_positions = self._get_owned_positions(board, agent_player)
        if not agent_positions:
            return 0.0

        protected_count = sum(1 for pos in agent_positions if self._is_in_moulin(board, pos, agent_player))

        return protected_count / len(agent_positions)

    def _calculate_blocked_piece_reward(
        self,
        board: "jdm_ru.PyBoard",
        agent_player: int,
    ) -> float:
        """
        Calculate reward based on the ratio of opponent's blocked pieces.
        Only applies during the Moving phase.

        Args:
            board: Current board state
            agent_player: The agent's player ID

        Returns:
            Ratio of blocked opponent pieces (0.0 to 1.0) in Moving phase, 0.0 otherwise
        """
        # Get phase from board state embedding
        phase = self._get_phase_from_board(board)

        if phase != 1:  # 1 = Moving phase
            return 0.0

        # Get opponent positions
        opponent = -agent_player
        opponent_positions = self._get_owned_positions(board, opponent)
        if not opponent_positions:
            return 0.0

        # Get all legal moves
        legal_moves = board.legal_moves()

        # Count how many opponent pieces have no legal moves
        blocked_count = 0
        for pos in opponent_positions:
            # Check if any legal move starts from this position
            has_move = any(move.from_position() == pos for move in legal_moves)
            if not has_move:
                blocked_count += 1

        return blocked_count / len(opponent_positions)

    def _normalize_reward(
        self,
        reward: float,
        board: "jdm_ru.PyBoard",
    ) -> float:
        """
        Normalize reward based on game phase to keep values in reasonable range.

        Args:
            reward: Raw reward value
            board: Board state for determining phase

        Returns:
            Normalized reward (typically in range [-5, 5])
        """
        phase = self._get_phase_from_board(board)

        # Define reward ranges per phase (from Rust implementation)
        if phase == 0:  # Placing
            reward_max, reward_min = 13.0, -14.0
        elif phase == 1:  # Moving
            reward_max, reward_min = 15.0, -14.0
        else:  # Flying (phase == 2)
            reward_max, reward_min = 13.0, -14.0

        # Normalize to approximately [-5, 5] range
        return 5.0 * (reward / (reward_max - reward_min))

    # Helper methods

    def _get_phase_from_board(self, board: "jdm_ru.PyBoard") -> int:
        """
        Extract phase from board embedding.

        Returns:
            0 = Placing, 1 = Moving, 2 = Flying
        """
        embedding = board.to_embed()
        # Phase is encoded at indices 2-4 (one-hot)
        if embedding[2] > 0.5:
            return 0  # Placing
        elif embedding[3] > 0.5:
            return 1  # Moving
        else:
            return 2  # Flying

    def _get_owned_positions(
        self,
        board: "jdm_ru.PyBoard",
        player: int,
    ) -> list[int]:
        """
        Get all board positions owned by a player.

        Args:
            board: Board state
            player: Player ID (1 or -1)

        Returns:
            List of position indices (0-23)
        """
        embedding = board.to_embed()
        positions = []

        # Board squares are at indices 5-76 (24 positions x 3 states each)
        # Format for each position: [white, black, empty]
        for pos in range(24):
            base_idx = 5 + (pos * 3)
            if (player == 1 and embedding[base_idx] > 0.5) or (player == -1 and embedding[base_idx + 1] > 0.5):  # White
                positions.append(pos)

        return positions

    def _forms_moulin(
        self,
        board: "jdm_ru.PyBoard",
        move: "jdm_ru.PyMove",
        player: int,
    ) -> bool:
        """
        Check if a move forms a moulin (mill) for a player.

        We simulate the move on the board and check for moulins.
        Note: This is a simplified check - ideally we'd have a Rust method for this.

        Args:
            board: Board state
            move: Move to check
            player: Player ID

        Returns:
            True if move forms a moulin
        """
        # Apply move to get new board state
        new_board = board.apply_move(move)

        # Get the destination position
        to_pos = move.to_position()

        # Check if destination is part of a moulin
        return self._is_in_moulin(new_board, to_pos, player)

    def _is_in_moulin(
        self,
        board: "jdm_ru.PyBoard",
        position: int,
        player: int,
    ) -> bool:
        """
        Check if a position is part of a moulin (mill).

        A moulin is three pieces in a row (horizontal or vertical).

        Args:
            board: Board state
            position: Position to check (0-23)
            player: Player ID

        Returns:
            True if position is in a moulin
        """
        # Mill patterns (from board constants)
        # Each mill is a list of 3 positions that form a line
        MOULINS = [
            # Outer square
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 0],
            # Middle square
            [8, 9, 10],
            [10, 11, 12],
            [12, 13, 14],
            [14, 15, 8],
            # Inner square
            [16, 17, 18],
            [18, 19, 20],
            [20, 21, 22],
            [22, 23, 16],
            # Connections
            [1, 9, 17],
            [3, 11, 19],
            [5, 13, 21],
            [7, 15, 23],
        ]

        # Get board state
        embedding = board.to_embed()

        # Check each moulin pattern
        for moulin in MOULINS:
            if position not in moulin:
                continue

            # Check if all three positions in this moulin belong to the player
            all_match = True
            for pos in moulin:
                base_idx = 5 + (pos * 3)
                if player == 1:  # White
                    if embedding[base_idx] < 0.5:
                        all_match = False
                        break
                else:  # Black
                    if embedding[base_idx + 1] < 0.5:
                        all_match = False
                        break

            if all_match:
                return True

        return False
