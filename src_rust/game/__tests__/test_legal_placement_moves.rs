#[cfg(test)]
mod tests {
    use crate::game::{Board, Move, Player};

    #[test]
    fn test_legal_placement_moves_empty_board() {
        let board = Board::new();
        let moves = board.legal_placement_moves();

        assert_eq!(moves.len(), 24);
        for i in 0..24 {
            assert!(moves.iter().any(|m| m.from_position.is_none()
                && m.to_position == i
                && m.removed_position.is_none()));
        }
    }

    #[test]
    fn test_legal_placement_moves_partially_filled_board() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[1] = Player::Black;
        board.squares[5] = Player::White;

        let moves = board.legal_placement_moves();

        assert_eq!(moves.len(), 21);
        assert!(!moves.iter().any(|m| m.to_position == 0));
        assert!(!moves.iter().any(|m| m.to_position == 1));
        assert!(!moves.iter().any(|m| m.to_position == 5));
        assert!(moves
            .iter()
            .any(|m| m.to_position == 2 && m.removed_position.is_none()));
        assert!(moves
            .iter()
            .any(|m| m.to_position == 3 && m.removed_position.is_none()));
    }

    #[test]
    fn test_legal_placement_moves_forms_moulin_horizontal() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[1] = Player::White;
        board.squares[3] = Player::Black;

        let moves = board.legal_placement_moves();
        let removal_move = moves
            .iter()
            .find(|m| m.to_position == 2 && m.removed_position.is_some());

        assert!(removal_move.is_some());

        let removal_move = removal_move.unwrap();
        assert!(removal_move.from_position.is_none());
        assert_eq!(removal_move.to_position, 2);
        assert_eq!(removal_move.removed_position, Some(3));
    }

    #[test]
    fn test_legal_placement_moves_forms_moulin_vertical() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[7] = Player::White;
        board.squares[1] = Player::Black;
        board.squares[14] = Player::Black;

        assert!(board.forms_moulin(&Move::placement(6), Player::White));

        let moves = board.legal_placement_moves();
        let removal_move = moves
            .iter()
            .find(|m| m.to_position == 6 && m.removed_position.is_some());

        assert!(removal_move.is_some());

        let removal_move = removal_move.unwrap();
        assert!(removal_move.from_position.is_none());
        assert_eq!(removal_move.to_position, 6);
        assert_eq!(removal_move.removed_position, Some(1));
    }

    #[test]
    fn test_cant_remove_piece_in_moulin() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[1] = Player::White;
        board.squares[2] = Player::White;
        board.squares[7] = Player::White;
        board.squares[10] = Player::Black;
        board.squares[11] = Player::Black;
        board.current_player = Player::Black;

        let moves = board.legal_placement_moves();
        let moulin_moves_to_12: Vec<_> = moves.iter().filter(|m| m.to_position == 12).collect();

        assert!(!moulin_moves_to_12.is_empty());
        assert_eq!(moulin_moves_to_12.len(), 1);

        let moulin_move_to_12 = moulin_moves_to_12[0];
        assert_eq!(moulin_move_to_12.to_position, 12);
        assert_eq!(moulin_move_to_12.removed_position, Some(7));
    }

    #[test]
    fn test_legal_placement_moves_opponent_all_in_moulin() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[1] = Player::White;
        board.squares[2] = Player::White;
        board.squares[10] = Player::Black;
        board.squares[11] = Player::Black;
        board.current_player = Player::Black;

        let moves = board.legal_placement_moves();
        let moulin_moves_to_12: Vec<_> = moves.iter().filter(|m| m.to_position == 12).collect();

        assert!(!moulin_moves_to_12.is_empty());
        assert_eq!(moulin_moves_to_12.len(), 3);

        for m in moulin_moves_to_12 {
            assert_eq!(m.to_position, 12);
            assert!(
                m.removed_position == Some(0)
                    || m.removed_position == Some(1)
                    || m.removed_position == Some(2)
            );
        }
    }

    #[test]
    fn test_legal_placement_moves_fully_occupied_board() {
        let mut board = Board::new();
        for i in 0..24 {
            board.squares[i] = if i % 2 == 0 {
                Player::White
            } else {
                Player::Black
            };
        }

        let moves = board.legal_placement_moves();

        assert_eq!(moves.len(), 0);
    }
}
