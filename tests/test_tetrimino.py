"""Tests for Tetrimino class."""

import pytest

from tetris.core.tetrimino import Tetrimino


class TestTetrimino:
    """Test Tetrimino class."""

    def test_create_tetrimino_with_shape(self):
        """Test creating a tetrimino with a specific shape."""
        tet = Tetrimino("I", x=5, y=10)
        assert tet.x == 5
        assert tet.y == 10
        assert tet.orientation == "N"
        assert tet.size == 4  # I piece is 4x4
        assert len(tet.minos) == 4
        # Verify it's the I piece by checking minos
        assert tet.minos[2] == ["c", "c", "c", "c"]  # I piece has cyan in row 2

    def test_get_minos(self):
        """Test getting minos for a shape."""
        minos = Tetrimino.get_minos("O")
        assert minos == [["y", "y"], ["y", "y"]]

        minos = Tetrimino.get_minos("I")
        assert len(minos) == 4
        assert minos[2] == ["c", "c", "c", "c"]  # I piece has cyan in row 2

    def test_get_minos_case_insensitive(self):
        """Test that get_minos handles case-insensitive shapes."""
        minos_lower = Tetrimino.get_minos("o")
        minos_upper = Tetrimino.get_minos("O")
        assert minos_lower == minos_upper

    def test_rand_shape(self):
        """Test getting a random shape."""
        shape = Tetrimino.rand_shape()
        assert shape in Tetrimino.shapes.keys()

        # Test multiple calls return valid shapes
        shapes = [Tetrimino.rand_shape() for _ in range(10)]
        assert all(s in Tetrimino.shapes.keys() for s in shapes)

    def test_make_bag(self):
        """Test creating a bag of tetrimino shapes."""
        bag = Tetrimino.make_bag(1)
        assert len(bag) == len(Tetrimino.shapes)
        assert set(bag) == set(Tetrimino.shapes.keys())

        # Test with repetitions
        bag = Tetrimino.make_bag(2)
        assert len(bag) == 2 * len(Tetrimino.shapes)
        # Each shape should appear twice
        for shape in Tetrimino.shapes.keys():
            assert bag.count(shape) == 2

    def test_make_bag_is_shuffled(self):
        """Test that make_bag shuffles the bag."""
        # Create multiple bags and check they're not all identical
        bags = [Tetrimino.make_bag(1) for _ in range(5)]
        # At least two should be different (very unlikely all 5 are same order)
        unique_bags = set(tuple(bag) for bag in bags)
        assert len(unique_bags) > 1

    def test_reset_ldml(self):
        """Test resetting lock down movements left."""
        tet = Tetrimino("I", x=5, y=10, ldml=5)
        assert tet.ldml == 5
        tet.reset_LDML()
        assert tet.ldml == 15

    def test_coords(self):
        """Test getting coordinates string."""
        tet = Tetrimino("I", x=5, y=10)
        assert tet.coords() == "(5,10)"

    def test_initial_state(self):
        """Test initial state of tetrimino."""
        tet = Tetrimino("T", x=3, y=20, orientation="E", ldml=10)
        assert tet.x == 3
        assert tet.y == 20
        assert tet.orientation == "E"
        assert tet.ldml == 10
        assert tet.landed is False
        assert tet.landed_y == -1
        assert tet.ld_timer == -1
        assert tet.lowest_y == 20
        assert tet.size == 3  # T piece is 3x3

    def test_all_shapes_exist(self):
        """Test that all standard Tetris shapes are defined."""
        expected_shapes = {"O", "I", "T", "L", "J", "S", "Z"}
        assert set(Tetrimino.shapes.keys()) == expected_shapes

    def test_shape_minos_are_valid(self):
        """Test that all shape minos are valid 2D lists."""
        for shape_name, minos in Tetrimino.shapes.items():
            assert isinstance(minos, list)
            assert len(minos) > 0
            for row in minos:
                assert isinstance(row, list)
                assert len(row) > 0
                # All rows should have same length
                assert len(row) == len(minos[0])
