#!/usr/bin/env python3
"""Test script to verify Move.to_indices() works correctly."""

import jdm_ru


def test_to_indices():
    """Test the new to_indices() method on various move types."""

    print("Testing Move.to_indices() method...")
    print("=" * 60)

    # Create a board and get some legal moves
    board = jdm_ru.PyBoard()
    legal_moves = board.legal_moves()

    if not legal_moves:
        print("❌ No legal moves found!")
        return False

    print(f"\n✅ Found {len(legal_moves)} legal moves")
    print("\nTesting first 5 moves:\n")

    for i, move in enumerate(legal_moves[:5]):
        # Get both representations
        indices = move.to_indices()
        embed = move.to_embed()

        # Display move details
        print(f"Move {i + 1}:")
        print(f"  Repr:       {move}")
        print(f"  Indices:    {indices}")
        print(f"  Type check: {type(indices)} with {len(indices)} elements")

        # Verify format
        assert len(indices) == 3, f"Expected 3 indices, got {len(indices)}"
        assert all(isinstance(idx, int) for idx in indices), "All indices should be integers"
        assert all(0 <= idx <= 24 for idx in indices), "All indices should be 0-24"

        # Verify indices match move attributes
        from_pos = move.from_position()
        to_pos = move.to_position()
        removed_pos = move.removed_position()

        expected_from = 24 if from_pos is None else from_pos
        expected_to = to_pos
        expected_removed = 24 if removed_pos is None else removed_pos

        assert indices[0] == expected_from, f"from_position mismatch: {indices[0]} != {expected_from}"
        assert indices[1] == expected_to, f"to_position mismatch: {indices[1]} != {expected_to}"
        assert indices[2] == expected_removed, f"removed_position mismatch: {indices[2]} != {expected_removed}"

        print("  ✅ Validation passed!")
        print()

    print("=" * 60)
    print("✅ All tests passed!")
    print("\n📊 Statistics:")
    print(f"  - Embedding size (old): {len(embed)} floats")
    print(f"  - Indices size (new):   {len(indices)} integers")
    print(f"  - Reduction factor:     {len(embed) / len(indices):.1f}x smaller")
    print("\n🎉 to_indices() is working correctly!")

    return True


if __name__ == "__main__":
    try:
        test_to_indices()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
