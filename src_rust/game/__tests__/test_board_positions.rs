#[cfg(test)]
mod tests {
    use crate::game::{Board, Move, Player};

    #[test]
    fn test_free_positions_empty_board() {
        let board = Board::new();
        let free_positions = board.free_positions();

        assert_eq!(free_positions.len(), 24);
        assert_eq!(free_positions, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn test_free_positions_partially_filled_board() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[1] = Player::Black;
        board.squares[5] = Player::White;

        let free_positions = board.free_positions();

        assert_eq!(free_positions.len(), 21);
        assert!(!free_positions.contains(&0));
        assert!(!free_positions.contains(&1));
        assert!(!free_positions.contains(&5));
        assert!(free_positions.contains(&2));
        assert!(free_positions.contains(&3));
        assert!(free_positions.contains(&4));
        assert!(free_positions.contains(&6));
    }

    #[test]
    fn test_free_positions_fully_occupied_board() {
        let mut board = Board::new();
        for i in 0..24 {
            board.squares[i] = if i % 2 == 0 {
                Player::White
            } else {
                Player::Black
            };
        }

        let free_positions = board.free_positions();

        assert_eq!(free_positions.len(), 0);
    }

    #[test]
    fn test_owned_positions_white_empty_board() {
        let board = Board::new();
        let white_positions = board.owned_positions(Player::White);

        assert_eq!(white_positions.len(), 0);
    }

    #[test]
    fn test_owned_positions_white_with_pieces() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[5] = Player::White;
        board.squares[10] = Player::White;
        board.squares[15] = Player::Black;
        board.squares[20] = Player::Black;

        let white_positions = board.owned_positions(Player::White);

        assert_eq!(white_positions.len(), 3);
        assert!(white_positions.contains(&0));
        assert!(white_positions.contains(&5));
        assert!(white_positions.contains(&10));
        assert!(!white_positions.contains(&15));
        assert!(!white_positions.contains(&20));
    }

    #[test]
    fn test_rotate_move() {
        let mv = Move::move_piece(0, 1).with_removal(23);
        assert_eq!(mv.from_position, Some(0));
        assert_eq!(mv.to_position, 1);
        assert_eq!(mv.removed_position, Some(23));

        let rotated_mv = mv.rotate(1);
        assert_eq!(rotated_mv.from_position, Some(2));
        assert_eq!(rotated_mv.to_position, 3);
        assert_eq!(rotated_mv.removed_position, Some(17));

        let rotated_mv = rotated_mv.rotate(1);
        assert_eq!(rotated_mv.from_position, Some(4));
        assert_eq!(rotated_mv.to_position, 5);
        assert_eq!(rotated_mv.removed_position, Some(19));

        let rotated_mv = rotated_mv.rotate(1);
        assert_eq!(rotated_mv.from_position, Some(6));
        assert_eq!(rotated_mv.to_position, 7);
        assert_eq!(rotated_mv.removed_position, Some(21));

        let rotated_mv = rotated_mv.rotate(1);
        assert_eq!(rotated_mv.from_position, Some(0));
        assert_eq!(rotated_mv.to_position, 1);
        assert_eq!(rotated_mv.removed_position, Some(23));
    }

    #[test]
    fn test_rotate_board() {
        let mut board = Board::new();
        board.squares[0] = Player::White;
        board.squares[8] = Player::Black;
        board.squares[16] = Player::White;
        board.squares[23] = Player::Black;

        let rotated_board = board.rotate(1);
        assert_eq!(rotated_board.squares[2], Player::White);
        assert_eq!(rotated_board.squares[10], Player::Black);
        assert_eq!(rotated_board.squares[18], Player::White);
        assert_eq!(rotated_board.squares[17], Player::Black);

        let rotated_board = rotated_board.rotate(1);
        assert_eq!(rotated_board.squares[4], Player::White);
        assert_eq!(rotated_board.squares[12], Player::Black);
        assert_eq!(rotated_board.squares[20], Player::White);
        assert_eq!(rotated_board.squares[19], Player::Black);

        let rotated_board = rotated_board.rotate(1);
        assert_eq!(rotated_board.squares[6], Player::White);
        assert_eq!(rotated_board.squares[14], Player::Black);
        assert_eq!(rotated_board.squares[22], Player::White);
        assert_eq!(rotated_board.squares[21], Player::Black);

        let rotated_board = rotated_board.rotate(1);
        assert_eq!(rotated_board.squares[0], Player::White);
        assert_eq!(rotated_board.squares[8], Player::Black);
        assert_eq!(rotated_board.squares[16], Player::White);
        assert_eq!(rotated_board.squares[23], Player::Black);
    }
}
