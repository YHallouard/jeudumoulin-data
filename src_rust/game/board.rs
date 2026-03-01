use serde::{Deserialize, Serialize};

use super::constants::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    White = 1,
    Black = -1,
    None = 0,
}

impl Player {
    pub fn opponent(&self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
            Player::None => Player::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    Placing = 0,
    Moving = 1,
    Flying = 2,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Move {
    pub from_position: Option<usize>,
    pub to_position: usize,
    pub removed_position: Option<usize>,
}

impl Move {
    pub fn placement(position: usize) -> Self {
        Move {
            from_position: None,
            to_position: position,
            removed_position: None,
        }
    }

    pub fn move_piece(from_position: usize, to_position: usize) -> Self {
        Move {
            from_position: Some(from_position),
            to_position,
            removed_position: None,
        }
    }

    pub fn with_removal(&self, removed_position: usize) -> Self {
        Move {
            from_position: self.from_position,
            to_position: self.to_position,
            removed_position: Some(removed_position),
        }
    }

    pub fn rotate(&self, times: usize) -> Self {
        let rotated_from_position = self.from_position.map(|from_pos| {
            LEVEL_SIZE * (from_pos / LEVEL_SIZE) + (from_pos + 2 * times) % LEVEL_SIZE
        });
        let rotated_to_position = LEVEL_SIZE * (self.to_position / LEVEL_SIZE)
            + (self.to_position + 2 * times) % LEVEL_SIZE;
        let rotated_removed_position = self.removed_position.map(|removed_pos| {
            LEVEL_SIZE * (removed_pos / LEVEL_SIZE) + (removed_pos + 2 * times) % LEVEL_SIZE
        });

        Move {
            from_position: rotated_from_position,
            to_position: rotated_to_position,
            removed_position: rotated_removed_position,
        }
    }

    /// Convert move to a flat embedding vector
    /// Format: [from_position (24 one-hot), to_position (24 one-hot), removed_position (24 one-hot)]
    /// Total: 72 features
    pub fn to_embed(&self) -> Vec<f32> {
        let mut features = vec![0.0; 72];

        // One-hot encode from_position (positions 0-23)
        if let Some(from_pos) = self.from_position {
            if from_pos < BOARD_SIZE {
                features[from_pos] = 1.0;
            }
        }

        // One-hot encode to_position (positions 24-47)
        if self.to_position < BOARD_SIZE {
            features[24 + self.to_position] = 1.0;
        }

        // One-hot encode removed_position (positions 48-71)
        if let Some(removed_pos) = self.removed_position {
            if removed_pos < BOARD_SIZE {
                features[48 + removed_pos] = 1.0;
            }
        }

        features
    }

    /// Convert move to compact indices representation for DQN with embeddings
    /// Format: [from_position, to_position, removed_position]
    /// Each index: 0-23 for valid position, 24 for None
    /// Total: 3 integers (i64)
    ///
    /// This representation allows using nn.Embedding layers in PyTorch instead of
    /// one-hot encoding, enabling the network to learn dense position representations.
    pub fn to_indices(&self) -> Vec<i64> {
        vec![
            self.from_position.map(|p| p as i64).unwrap_or(24),
            self.to_position as i64,
            self.removed_position.map(|p| p as i64).unwrap_or(24),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pub squares: Vec<Player>,
    pub phase: Phase,
    pub white_unplaced: usize,
    pub black_unplaced: usize,
    pub current_player: Player,
    pub turn_counter: usize,
}

impl Board {
    pub fn new() -> Self {
        Board {
            squares: vec![Player::None; BOARD_SIZE],
            phase: Phase::Placing,
            white_unplaced: 9,
            black_unplaced: 9,
            current_player: Player::White,
            turn_counter: 0,
        }
    }

    pub fn piece_count(&self, player: Player) -> usize {
        self.squares.iter().filter(|&&p| p == player).count()
    }

    pub fn is_terminal(&self) -> bool {
        if self.phase == Phase::Placing {
            return false;
        }

        let white_pieces = self.piece_count(Player::White);
        let black_pieces = self.piece_count(Player::Black);

        if white_pieces < 3 || black_pieces < 3 {
            return true;
        }

        if self.phase == Phase::Moving {
            let moves = self.legal_moves();
            return moves.is_empty();
        }

        false
    }

    pub fn winner(&self) -> Option<Player> {
        let white_pieces = self.piece_count(Player::White);
        let black_pieces = self.piece_count(Player::Black);
        let is_terminal = self.is_terminal();
        let no_moves = self.legal_moves().is_empty();

        if is_terminal && no_moves {
            Some(self.current_player.opponent())
        } else if is_terminal && white_pieces < 3 {
            Some(Player::Black)
        } else if is_terminal && black_pieces < 3 {
            Some(Player::White)
        } else {
            None
        }
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        match self.phase {
            Phase::Placing => self.legal_placement_moves(),
            Phase::Moving => self.legal_move_moves(),
            Phase::Flying => self.legal_flying_moves(),
        }
    }

    pub fn owned_positions(&self, player: Player) -> Vec<usize> {
        self.squares
            .iter()
            .enumerate()
            .filter(|(_, &p)| p == player)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn opponent_all_in_moulin(&self) -> bool {
        let player_moulins: Vec<_> = MOULINS
            .iter()
            .filter(|moulin| {
                moulin
                    .iter()
                    .all(|&i| self.squares[i] == self.current_player.opponent())
            })
            .collect();

        let flattened_moulins: Vec<usize> = player_moulins
            .iter()
            .flat_map(|moulin| moulin.iter())
            .copied()
            .collect();

        let opponent_positions = self.owned_positions(self.current_player.opponent());

        opponent_positions
            .iter()
            .all(|pos| flattened_moulins.contains(pos))
    }

    pub fn free_positions(&self) -> Vec<usize> {
        self.squares
            .iter()
            .enumerate()
            .filter(|(_, &p)| p == Player::None)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn forms_moulin(&self, mov: &Move, player: Player) -> bool {
        let mut squares = self.squares.clone();
        squares[mov.to_position] = player;
        if let Some(from_pos) = mov.from_position {
            squares[from_pos] = Player::None;
        }

        let filtered_moulins: Vec<_> = MOULINS
            .iter()
            .filter(|moulin| moulin.contains(&mov.to_position))
            .collect();

        filtered_moulins
            .iter()
            .any(|moulin| moulin.iter().all(|&i| squares[i] == player))
    }

    pub fn get_player_not_protected_positions(
        &self,
        opponent_positions: &[usize],
        _player: Player,
    ) -> Vec<usize> {
        let owned_moulins: Vec<_> = MOULINS
            .iter()
            .filter(|moulin| moulin.iter().all(|pos| opponent_positions.contains(pos)))
            .collect();

        let flattened_owned_moulins: Vec<usize> = owned_moulins
            .iter()
            .flat_map(|moulin| moulin.iter())
            .copied()
            .collect();

        opponent_positions
            .iter()
            .filter(|pos| !flattened_owned_moulins.contains(pos))
            .copied()
            .collect()
    }

    pub fn generate_removal_moves(
        &self,
        base: &Move,
        opponent_positions: &[usize],
        is_all_opponent_in_moulin: bool,
    ) -> Vec<Move> {
        if is_all_opponent_in_moulin {
            opponent_positions
                .iter()
                .map(|&pos| base.with_removal(pos))
                .collect()
        } else {
            self.get_player_not_protected_positions(
                opponent_positions,
                self.current_player.opponent(),
            )
            .iter()
            .map(|&pos| base.with_removal(pos))
            .collect()
        }
    }

    pub fn legal_placement_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        let opponent_positions = self.owned_positions(self.current_player.opponent());
        let is_all_opponent_in_moulin = self.opponent_all_in_moulin();

        for position in self.free_positions() {
            let base = Move::placement(position);
            if self.forms_moulin(&base, self.current_player) {
                let removal_moves = self.generate_removal_moves(
                    &base,
                    &opponent_positions,
                    is_all_opponent_in_moulin,
                );
                moves.extend(removal_moves);
            } else {
                moves.push(base);
            }
        }
        moves
    }

    pub fn legal_move_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        let opponent_positions = self.owned_positions(self.current_player.opponent());
        let is_all_opponent_in_moulin = self.opponent_all_in_moulin();

        for position in self.owned_positions(self.current_player) {
            for &neighbor in NEIGHBORS[position] {
                if self.squares[neighbor] == Player::None {
                    let base = Move::move_piece(position, neighbor);
                    if self.forms_moulin(&base, self.current_player) {
                        let removal_moves = self.generate_removal_moves(
                            &base,
                            &opponent_positions,
                            is_all_opponent_in_moulin,
                        );
                        moves.extend(removal_moves);
                    } else {
                        moves.push(base);
                    }
                }
            }
        }
        moves
    }

    pub fn legal_flying_moves(&self) -> Vec<Move> {
        let owned_positions = self.owned_positions(self.current_player);

        if owned_positions.len() > 3 {
            return self.legal_move_moves();
        }

        let mut moves = Vec::new();
        let opponent_positions = self.owned_positions(self.current_player.opponent());
        let is_all_opponent_in_moulin = self.opponent_all_in_moulin();

        for &from_position in &owned_positions {
            for position in self.free_positions() {
                let base = Move::move_piece(from_position, position);
                if self.forms_moulin(&base, self.current_player) {
                    let removal_moves = self.generate_removal_moves(
                        &base,
                        &opponent_positions,
                        is_all_opponent_in_moulin,
                    );
                    moves.extend(removal_moves);
                } else {
                    moves.push(base);
                }
            }
        }
        moves
    }

    pub fn switch_phase(&mut self) {
        match self.phase {
            Phase::Placing => {
                if self.white_unplaced == 0 && self.black_unplaced == 0 {
                    self.phase = Phase::Moving;
                }
            }
            Phase::Moving => {
                if self.piece_count(self.current_player) == 3
                    || self.piece_count(self.current_player.opponent()) == 3
                {
                    self.phase = Phase::Flying;
                }
            }
            Phase::Flying => {}
        }
    }

    pub fn move_from_to(
        &mut self,
        from_position: usize,
        to_position: usize,
        removed_position: Option<usize>,
    ) -> Result<(), String> {
        if from_position >= BOARD_SIZE || to_position >= BOARD_SIZE {
            return Err("Invalid move: position out of bounds".to_string());
        }
        if let Some(removed_pos) = removed_position {
            if removed_pos >= BOARD_SIZE {
                return Err("Invalid move: removed position out of bounds".to_string());
            }
        }
        if self.squares[from_position] != self.current_player {
            return Err("Invalid move: not your piece".to_string());
        }
        if self.squares[to_position] != Player::None {
            return Err("Invalid move: destination occupied".to_string());
        }
        if let Some(removed_pos) = removed_position {
            if self.squares[removed_pos] != self.current_player.opponent() {
                return Err("Invalid move: cannot remove that piece".to_string());
            }
        }
        if self.phase == Phase::Moving && !NEIGHBORS[from_position].contains(&to_position) {
            return Err("Invalid move: not a neighbor".to_string());
        }

        self.squares[from_position] = Player::None;
        self.squares[to_position] = self.current_player;
        if let Some(removed_pos) = removed_position {
            self.squares[removed_pos] = Player::None;
        }

        Ok(())
    }

    pub fn move_to(
        &mut self,
        to_position: usize,
        removed_position: Option<usize>,
    ) -> Result<(), String> {
        if to_position >= BOARD_SIZE {
            return Err("Invalid move: position out of bounds".to_string());
        }
        if let Some(removed_pos) = removed_position {
            if removed_pos >= BOARD_SIZE {
                return Err("Invalid move: removed position out of bounds".to_string());
            }
        }
        if self.squares[to_position] != Player::None {
            return Err("Invalid move: destination occupied".to_string());
        }
        if let Some(removed_pos) = removed_position {
            if self.squares[removed_pos] != self.current_player.opponent() {
                return Err("Invalid move: cannot remove that piece".to_string());
            }
        }
        if (self.current_player == Player::White && self.white_unplaced == 0)
            || (self.current_player == Player::Black && self.black_unplaced == 0)
        {
            return Err("Invalid move: no unplaced pieces".to_string());
        }

        self.squares[to_position] = self.current_player;
        if let Some(removed_pos) = removed_position {
            self.squares[removed_pos] = Player::None;
        }

        Ok(())
    }

    pub fn apply_move(&self, mov: &Move) -> Board {
        let mut board = self.clone();

        if let Some(from_pos) = mov.from_position {
            board
                .move_from_to(from_pos, mov.to_position, mov.removed_position)
                .ok();
        } else {
            if self.phase != Phase::Placing {
                let _ = board.legal_moves();
            }
            board.move_to(mov.to_position, mov.removed_position).ok();
        }

        if self.phase == Phase::Placing {
            if self.current_player == Player::White {
                board.white_unplaced = board.white_unplaced.saturating_sub(1);
            } else {
                board.black_unplaced = board.black_unplaced.saturating_sub(1);
            }
        }

        board.current_player = self.current_player.opponent();
        board.switch_phase();
        board.turn_counter += 1;
        board
    }

    pub fn rotate(&self, times: usize) -> Board {
        let mut new_board = Board::new();
        let mut rotated_squares = vec![Player::None; BOARD_SIZE];

        for i in 0..BOARD_SIZE {
            let rotated_index = LEVEL_SIZE * (i / LEVEL_SIZE) + (i + 2 * times) % LEVEL_SIZE;
            rotated_squares[rotated_index] = self.squares[i];
        }

        new_board.squares = rotated_squares;
        new_board.phase = self.phase;
        new_board.white_unplaced = self.white_unplaced;
        new_board.black_unplaced = self.black_unplaced;
        new_board.current_player = self.current_player;
        new_board.turn_counter = self.turn_counter;
        new_board
    }

    /// Convert board state to a flat embedding vector for neural network input
    /// Format: [current_player(2), phase(3), board_squares(24*3), unplaced_counts(2)]
    /// Total: 77 features
    pub fn to_embed(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(77);

        // One-hot encode current player (2 features)
        features.push(if self.current_player == Player::White {
            1.0
        } else {
            0.0
        });
        features.push(if self.current_player == Player::Black {
            1.0
        } else {
            0.0
        });

        // One-hot encode phase (3 features)
        features.push(if self.phase == Phase::Placing {
            1.0
        } else {
            0.0
        });
        features.push(if self.phase == Phase::Moving {
            1.0
        } else {
            0.0
        });
        features.push(if self.phase == Phase::Flying {
            1.0
        } else {
            0.0
        });

        // One-hot encode each square (24 * 3 = 72 features)
        for square in &self.squares {
            features.push(if *square == Player::White { 1.0 } else { 0.0 });
            features.push(if *square == Player::Black { 1.0 } else { 0.0 });
            features.push(if *square == Player::None { 1.0 } else { 0.0 });
        }

        features
    }

    pub fn print_board(&self) {
        let get_piece = |pos: usize| -> char {
            match self.squares[pos] {
                Player::White => 'W',
                Player::Black => 'B',
                Player::None => '.',
            }
        };

        println!(
            "{}----------{}----------{}",
            get_piece(0),
            get_piece(1),
            get_piece(2)
        );
        println!("|          |          |");
        println!(
            "|  {}-------{}-------{}  |",
            get_piece(8),
            get_piece(9),
            get_piece(10)
        );
        println!("|  |       |       |  |");
        println!(
            "|  |   {}---{}---{}   |  |",
            get_piece(16),
            get_piece(17),
            get_piece(18)
        );
        println!("|  |   |       |   |  |");
        println!(
            "{}---{}---{}       {}---{}--{}",
            get_piece(7),
            get_piece(15),
            get_piece(23),
            get_piece(19),
            get_piece(11),
            get_piece(3)
        );
        println!("|  |   |       |   |  |");
        println!(
            "|  |   {}---{}---{}   |  |",
            get_piece(22),
            get_piece(21),
            get_piece(20)
        );
        println!("|  |       |       |  |");
        println!(
            "|  {}-------{}-------{}  |",
            get_piece(14),
            get_piece(13),
            get_piece(12)
        );
        println!("|          |          |");
        println!(
            "{}----------{}----------{}",
            get_piece(6),
            get_piece(5),
            get_piece(4)
        );
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}
