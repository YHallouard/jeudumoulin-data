#[cfg(test)]
mod tests {
    use crate::game::{Board, Phase, Player};

    #[test]
    fn test_legal_moving_moves_empty_board() {
        let board = Board::new();
        let moves = board.legal_move_moves();

        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_legal_moving_moves_single_piece_no_neighbors() {
        let mut board = Board::new();
        board.phase = Phase::Moving;
        board.squares[0] = Player::White;

        let moves = board.legal_move_moves();

        assert_eq!(moves.len(), 2);

        let move_to_1 = moves
            .iter()
            .find(|m| m.from_position == Some(0) && m.to_position == 1);
        let move_to_7 = moves
            .iter()
            .find(|m| m.from_position == Some(0) && m.to_position == 7);

        assert!(move_to_1.is_some());
        assert!(move_to_7.is_some());

        let move_to_1 = move_to_1.unwrap();
        assert_eq!(move_to_1.from_position, Some(0));
        assert_eq!(move_to_1.to_position, 1);
        assert!(move_to_1.removed_position.is_none());

        let move_to_7 = move_to_7.unwrap();
        assert_eq!(move_to_7.from_position, Some(0));
        assert_eq!(move_to_7.to_position, 7);
        assert!(move_to_7.removed_position.is_none());
    }

    #[test]
    fn test_legal_moving_moves_single_piece_with_blocked_neighbors() {
        let mut board = Board::new();
        board.phase = Phase::Moving;
        board.squares[0] = Player::White;
        board.squares[1] = Player::Black;
        board.squares[7] = Player::Black;

        let moves = board.legal_move_moves();

        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_legal_moving_moves_forms_moulin_horizontal() {
        let mut board = Board::new();
        board.phase = Phase::Moving;
        board.squares[1] = Player::White;
        board.squares[2] = Player::White;
        board.squares[7] = Player::White;
        board.squares[9] = Player::Black;
        board.squares[8] = Player::Black;
        board.squares[15] = Player::Black;
        board.current_player = Player::White;

        let moves = board.legal_move_moves();

        let move_1_to_0 = moves
            .iter()
            .find(|m| m.from_position == Some(1) && m.to_position == 0);
        let move_7_to_0 = moves.iter().find(|m| {
            m.from_position == Some(7) && m.to_position == 0 && m.removed_position.is_some()
        });

        assert!(move_1_to_0.is_some());
        let move_1_to_0 = move_1_to_0.unwrap();
        assert_eq!(move_1_to_0.from_position, Some(1));
        assert_eq!(move_1_to_0.to_position, 0);
        assert!(move_1_to_0.removed_position.is_none());

        assert!(move_7_to_0.is_some());
        let move_7_to_0 = move_7_to_0.unwrap();
        assert_eq!(move_7_to_0.from_position, Some(7));
        assert_eq!(move_7_to_0.to_position, 0);
        assert_eq!(move_7_to_0.removed_position, Some(8));
    }

    #[test]
    fn test_legal_moving_moves_opponent_all_in_moulin() {
        let mut board = Board::new();
        board.phase = Phase::Moving;
        board.squares[1] = Player::White;
        board.squares[2] = Player::White;
        board.squares[7] = Player::White;
        board.squares[8] = Player::Black;
        board.squares[9] = Player::Black;
        board.squares[10] = Player::Black;

        let moves = board.legal_move_moves();

        let removal_move_1 = moves.iter().find(|m| {
            m.from_position == Some(7) && m.to_position == 0 && m.removed_position.is_some()
        });

        assert!(removal_move_1.is_some());

        let removal_move_1 = removal_move_1.unwrap();
        assert_eq!(removal_move_1.from_position, Some(7));
        assert_eq!(removal_move_1.to_position, 0);
        assert!(removal_move_1.removed_position.is_some());
    }
}
