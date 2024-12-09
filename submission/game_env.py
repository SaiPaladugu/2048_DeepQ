import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=4, goal=2**17):
        '''
        Initialize the 2048 game environment.

        Parameters:
        - size (int): The size of the game board (size x size).
        - goal (int): The target tile value to reach (default is 2048).
        '''
        super(Game2048Env, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(8)  # 8 possible actions
        self.observation_space = spaces.Box(
            low=0,
            high=goal,
            shape=(self.size, self.size),
            dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        '''
        Reset the game to its initial state.

        Parameters:
        - seed (int): Random seed for reproducibility.
        - options (dict): Additional options.

        Returns:
        - observation (ndarray): The initial board state.
        - info (dict): Additional information.
        '''
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        # Start with two random tiles
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy(), {}

    def step(self, action):
        '''
        Perform an action in the environment.

        Parameters:
        - action (int): The action to perform.

        Actions:
        0: Up
        1: Down
        2: Left
        3: Right
        4: Up Left (Diagonal)
        5: Up Right (Diagonal)
        6: Down Left (Diagonal)
        7: Down Right (Diagonal)

        Returns:
        - observation (ndarray): The new board state.
        - reward (int): The score obtained from this action.
        - done (bool): Whether the game is over.
        - info (dict): Additional information.
        '''
        moved = False
        score = 0
        if action == 0:
            # Up
            self.board = np.rot90(self.board, 1)
            moved, score = self.move()
            self.board = np.rot90(self.board, -1)

        elif action == 1:
            # Down
            self.board = np.rot90(self.board, -1)
            moved, score = self.move()
            self.board = np.rot90(self.board, 1)

        elif action == 2:
            # Left
            moved, score = self.move()

        elif action == 3:
            # Right
            self.board = np.fliplr(self.board)
            moved, score = self.move()
            self.board = np.fliplr(self.board)

        elif action == 4:
            # Up Left (Diagonal)
            self.board = self.rot45(self.board, 1)
            moved, score = self.move(diagonal=True)
            self.board = self.rot45(self.board, -1)

        elif action == 5:
            # Up Right (Diagonal)
            self.board = np.fliplr(self.board)
            self.board = self.rot45(self.board, 1)
            moved, score = self.move(diagonal=True)
            self.board = self.rot45(self.board, -1)
            self.board = np.fliplr(self.board)

        elif action == 6:
            # Down Left (Diagonal)
            self.board = self.rot45(self.board, -1)
            moved, score = self.move(diagonal=True)
            self.board = self.rot45(self.board, 1)

        elif action == 7:
            # Down Right (Diagonal)
            self.board = np.fliplr(self.board)
            self.board = self.rot45(self.board, -1)
            moved, score = self.move(diagonal=True)
            self.board = self.rot45(self.board, 1)
            self.board = np.fliplr(self.board)

        else:
            raise ValueError("Invalid action")

        if moved:
            self.add_random_tile()

        done = not self.can_move()
        reward = score
        info = {'moved': moved}
        return self.board.copy(), reward, done, False, info

    def render(self):
        '''
        Render the current state of the board.
        '''
        print(self.board)

    def add_random_tile(self):
        '''
        Add a new tile to the board at a random empty position.

        The new tile is either a 2 (90% chance) or a 4 (10% chance).
        '''
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[row][col] = 2 if np.random.random() < 0.9 else 4

    def can_move(self):
        '''
        Check if any moves are possible.

        A move is possible if:
        - There is at least one empty cell.
        - Two adjacent tiles in any direction (including diagonals) can be merged.
        '''
        if np.any(self.board == 0):
            return True
        for row in range(self.size):
            for col in range(self.size - 1):
                if self.board[row][col] == self.board[row][col + 1]:
                    return True
        for col in range(self.size):
            for row in range(self.size - 1):
                if self.board[row][col] == self.board[row + 1][col]:
                    return True
        # Check diagonals (top-left to bottom-right)
        tl_br_diag_matrix = self.rot45(self.board, 1)
        for row in tl_br_diag_matrix:
            for col in range(len(row) - 1):
                if row[col] == row[col+1]:
                    return True
        # Check diagonals (top-right to bottom-left)
        tr_bl_diag_matrix = self.rot45(self.board, -1)
        for row in tr_bl_diag_matrix:
            for col in range(len(row) - 1):
                if row[col] == row[col+1]:
                    return True
        return False

    def move(self, diagonal=False):
        '''
        Move the tiles in the specified direction.

        Parameters:
        - diagonal (bool): Whether the move is diagonal or not.

        Returns:
        - moved (bool): Whether any tiles moved.
        - score (int): The score obtained from merging tiles.
        '''
        if not diagonal:
            return self.move_square()
        else:
            return self.move_rotated()

    def move_square(self):
        '''
        Perform a move in one of the cardinal directions (up, down, left, right).

        This method slides and merges tiles in the board for cardinal moves.

        Returns:
        - moved (bool): Whether any tiles moved.
        - score (int): The score obtained from merging tiles.
        '''
        moved = False
        score = 0
        new_board = self.board.copy()
        for row in range(self.size):
            # Extract non-zero tiles
            tiles = self.board[row][self.board[row] != 0]
            merged_tiles = []
            skip = False
            i = 0
            while i < len(tiles):
                if skip:
                    skip = False
                    i += 1
                    continue
                if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                    # Merge tiles if two adjacent tiles are equal
                    merged_tiles.append(tiles[i] * 2)
                    score += tiles[i] * 2
                    skip = True  # Skip the next tile since it has been merged
                else:
                    merged_tiles.append(tiles[i])
                i += 1
            # Pad the merged tiles with zeros to maintain board size
            merged_tiles += [0] * (self.size - len(merged_tiles))
            if not np.array_equal(self.board[row], merged_tiles):
                moved = True
            new_board[row] = merged_tiles
        if moved:
            self.board = new_board
        return moved, score

    def move_rotated(self):
        '''
        Perform a move in one of the diagonal directions.

        This method slides and merges tiles in the board for diagonal moves.

        Returns:
        - moved (bool): Whether any tiles moved.
        - score (int): The score obtained from merging tiles.
        '''
        moved = False
        score = 0
        new_board = self.board.copy()  # Create a copy to store changes
        for row in range(len(self.board)):
            # Extract non-zero tiles
            tiles = self.board[row][self.board[row] != 0]
            merged_tiles = []
            skip = False
            i = 0
            while i < len(tiles):
                if skip:
                    skip = False
                    i += 1
                    continue
                if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                    # Merge tiles if two adjacent tiles are equal
                    merged_tiles.append(tiles[i] * 2)
                    score += tiles[i] * 2
                    skip = True  # Skip the next tile since it has been merged
                else:
                    merged_tiles.append(tiles[i])
                i += 1
            # Pad the merged tiles with zeros to maintain row length
            merged_tiles += [0] * (self.board[row].shape[0] - len(merged_tiles))
            # Check if any tile has changed in this row
            if not np.array_equal(self.board[row], merged_tiles):
                moved = True
            # Update the new_board with merged tiles
            new_board[row] = merged_tiles
        # Only update the actual board if a move has occurred
        if moved:
            self.board = new_board
        return moved, score

    def rot45(self, square_matrix_or_rotated_diagonals, direction):
        '''
        Rotate the board by 45 degrees.

        This function can either rotate a square matrix into its diagonals, or reconstruct a square matrix from its rotated diagonals.

        Parameters:
        - square_matrix_or_rotated_diagonals: Either the square matrix to rotate, or the rotated diagonals.
        - direction (int): 1 for counter-clockwise rotation, -1 for clockwise rotation.

        Returns:
        - Rotated diagonals or reconstructed square matrix, depending on the input.
        '''
        if isinstance(square_matrix_or_rotated_diagonals, np.ndarray):
            # Input is a square matrix; rotate it into diagonals
            return self.rot45_from_square(square_matrix_or_rotated_diagonals, direction)
        else:
            # Input is rotated diagonals; reconstruct the square matrix
            return self.rot45_to_square(square_matrix_or_rotated_diagonals, direction)

    def rot45_from_square(self, square_matrix, direction):
        '''
        Rotate a square matrix by 45 degrees, converting it into diagonals.

        Parameters:
        - square_matrix (ndarray): The square matrix to rotate.
        - direction (int): 1 for counter-clockwise rotation, -1 for clockwise rotation.

        Returns:
        - List of arrays: The rotated diagonals of the matrix.
        '''
        n, _ = square_matrix.shape
        # Determine the lengths of the diagonals after rotation
        diagonal_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
        rotated_diagonals = [np.ones(length, dtype=np.int64) for length in diagonal_lengths]
        if direction == 1:
            # Rotate the matrix by 90 degrees counter-clockwise if direction is 1
            square_matrix = np.rot90(square_matrix, 1)

        for i in range(n):
            for j in range(n):
                row = i + j
                element_index = j if row < n else n - 1 - i
                rotated_diagonals[row][element_index] = square_matrix[i, j]

        return rotated_diagonals

    def rot45_to_square(self, rotated_diagonals, direction):
        '''
        Convert rotated diagonals back into a square matrix.

        Parameters:
        - rotated_diagonals (list of arrays): The diagonals representing the rotated matrix.
        - direction (int): 1 for counter-clockwise rotation, -1 for clockwise rotation.

        Returns:
        - ndarray: The reconstructed square matrix.
        '''
        n = (len(rotated_diagonals) + 1) // 2
        square_matrix = np.zeros((n, n), dtype=np.int64)

        # Reassemble the square matrix from the rotated diagonals
        for row in range(len(rotated_diagonals)):
            for element_index, value in enumerate(rotated_diagonals[row]):
                if row < n:
                    i = row + element_index + 1 - len(rotated_diagonals[row])
                    j = n - 1 - row + element_index
                else:
                    i = row - n + 1 + element_index
                    j = element_index
                square_matrix[i, j] = value
        if direction == -1:
            return square_matrix
        else:
            # Rotate the matrix back by 90 degrees clockwise if direction is 1
            return np.rot90(square_matrix, 1)
    
    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        self.board = state.copy()
