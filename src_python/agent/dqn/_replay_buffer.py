import random
from collections import deque

import torch


class ExperienceReplayBuffer:
    """
    Circular buffer for storing and sampling experience tuples.

    The replay buffer stores transitions (s, a, r, s', done) and allows
    random sampling of batches for training. This provides two key benefits:

    1. **Breaks temporal correlation**: Consecutive game states are highly
       correlated, which can cause instability in neural network training.
       Random sampling decorrelates the training data.

    2. **Sample efficiency**: Each experience can be used multiple times
       for training, improving data efficiency.

    The buffer uses a deque with fixed maximum size, implementing a FIFO
    (First In, First Out) policy when the buffer is full.

    Args:
        max_size: Maximum number of transitions to store (default: 10,000)
        device: PyTorch device for tensors ('cpu' or 'cuda')

    Example:
        >>> buffer = ExperienceReplayBuffer(max_size=10000)
        >>> # Add experiences
        >>> state = [0.1, 0.2, ...]  # 77 floats
        >>> action = [5, 10, 24]     # 3 indices
        >>> reward = 0.5
        >>> next_state = [0.15, 0.25, ...]
        >>> done = False
        >>> buffer.add(state, action, reward, next_state, done)
        >>>
        >>> # Sample a batch for training
        >>> if len(buffer) >= 32:
        >>>     batch = buffer.sample(32)
        >>>     states, actions, rewards, next_states, dones = batch
    """

    def __init__(self, max_size: int = 10000, device: str = "cpu"):
        """
        Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store
            device: Device for PyTorch tensors ('cpu' or 'cuda')
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.device = device

    def add(
        self,
        state: list[float],
        action: list[int],
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: State embedding (77 floats)
            action: Action indices (3 ints: [from, to, remove])
            reward: Reward received
            next_state: Next state embedding (77 floats)
            done: Whether the episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors:
            - states: shape (batch_size, 77)
            - actions: shape (batch_size, 3)
            - rewards: shape (batch_size,)
            - next_states: shape (batch_size, 77)
            - dones: shape (batch_size,) as float (0.0 or 1.0)

        Raises:
            ValueError: If batch_size > len(buffer)
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with only {len(self.buffer)} transitions"
            )

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def can_sample(self, batch_size: int) -> bool:
        """
        Check if the buffer has enough transitions to sample a batch.

        Args:
            batch_size: Desired batch size

        Returns:
            True if buffer contains at least batch_size transitions
        """
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()

    def get_statistics(self) -> dict:
        """
        Get buffer statistics for monitoring.

        Returns:
            Dictionary with buffer statistics:
            - size: Current number of transitions
            - capacity: Maximum capacity
            - fill_ratio: Percentage of buffer filled
            - avg_reward: Average reward in buffer
            - terminal_ratio: Ratio of terminal states
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.max_size,
                "fill_ratio": 0.0,
                "avg_reward": 0.0,
                "terminal_ratio": 0.0,
            }

        rewards = [t[2] for t in self.buffer]
        dones = [t[4] for t in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.max_size,
            "fill_ratio": len(self.buffer) / self.max_size,
            "avg_reward": sum(rewards) / len(rewards),
            "terminal_ratio": sum(dones) / len(dones),
        }


class PrioritizedExperienceReplayBuffer(ExperienceReplayBuffer):
    """
    Prioritized Experience Replay Buffer.

    Extension of the standard replay buffer that samples transitions based
    on their TD-error (priority). Transitions with higher TD-error are
    sampled more frequently, which can accelerate learning by focusing on
    more "surprising" or informative transitions.

    This is an advanced feature and optional for basic DQN training.

    Reference:
        Schaul et al., "Prioritized Experience Replay", ICLR 2016
        https://arxiv.org/abs/1511.05952

    Args:
        max_size: Maximum number of transitions to store
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        beta_increment: Amount to increase beta per sample
        device: PyTorch device for tensors

    Note:
        This is a simplified implementation. For production use, consider
        using a sum tree data structure for O(log n) sampling complexity.
    """

    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        device: str = "cpu",
    ):
        """Initialize the prioritized replay buffer."""
        super().__init__(max_size, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=max_size)
        self.max_priority = 1.0

    def add(
        self,
        state: list[float],
        action: list[int],
        reward: float,
        next_state: list[float],
        done: bool,
        priority: float | None = None,
    ) -> None:
        """
        Add a transition with priority.

        Args:
            state: State embedding
            action: Action indices
            reward: Reward received
            next_state: Next state embedding
            done: Whether episode terminated
            priority: Priority value (if None, uses max_priority)
        """
        super().add(state, action, reward, next_state, done)

        if priority is None:
            priority = self.max_priority

        self.priorities.append(priority)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Sample a batch based on priorities.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
            where weights are importance sampling weights for correcting bias.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with only {len(self.buffer)} transitions"
            )

        priorities = torch.tensor(list(self.priorities), dtype=torch.float32)
        probabilities = priorities**self.alpha
        probabilities = probabilities / probabilities.sum()

        indices = torch.multinomial(probabilities, batch_size, replacement=False).tolist()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        N = len(self.buffer)
        weights = (N * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max for stability

        self.beta = min(1.0, self.beta + self.beta_increment)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_tensor = weights.to(self.device)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
            indices,
        )

    def update_priorities(self, indices: list[int], priorities: torch.Tensor) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD-errors)
        """
        priorities_list = priorities.detach().cpu().tolist()

        for idx, priority in zip(indices, priorities_list):
            self.priorities[idx] = abs(priority) + 1e-6  # Add small epsilon to avoid zero priority
            self.max_priority = max(self.max_priority, abs(priority))
