import gymnasium as gym
from gymnasium import spaces
import numpy as np

def rot45_from_square(square_matrix, direction):
    n, n = square_matrix.shape
    diagonal_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    rotated_diagonals = [np.ones(length, dtype=np.int64) for length in diagonal_lengths]
    if direction == 1: square_matrix = np.fliplr(square_matrix)
    
    # Fill in the diagonals based on the rotation direction
    for i in range(n):
        for j in range(n):
            row = i + j
            if direction == -1:  # Clockwise
                element_index = j if row < n else n - 1 - i
            elif direction == 1:  # Counterclockwise
                element_index = i if row < n else n - 1 - j
               
            rotated_diagonals[row][element_index] = square_matrix[i, j]
        
    return rotated_diagonals

def rot45_to_square(rotated_diagonals, direction):
    n = (len(rotated_diagonals) + 1) // 2
    square_matrix = np.zeros((n, n), dtype=np.int64)

    # Reassemble the square matrix from the rotated diagonals
    for row in range(len(rotated_diagonals)):
        for element_index, value in enumerate(rotated_diagonals[row]):
            if row < n:
                if direction == -1:  # Clockwise
                    i, j = element_index, row - element_index
                else:  # Counterclockwise
                    i, j = row - element_index, element_index
            else:
                if direction == -1:  # Clockwise
                    i, j = element_index - len(rotated_diagonals[row]), n - 1 - element_index
                else:  # Counterclockwise
                    i, j = n - 1 - element_index, element_index - len(rotated_diagonals[row])
            square_matrix[i, j] = value

    return np.fliplr(square_matrix) if direction == -1 else square_matrix

def rot45(square_matrix_or_rotated_diagonals, direction):
        # Initialize the result as a list of numpy arrays with the specified lengths
        if isinstance(square_matrix_or_rotated_diagonals, np.ndarray):
            return rot45_from_square(square_matrix_or_rotated_diagonals, direction)
        else:
            return rot45_to_square(square_matrix_or_rotated_diagonals, direction)

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=4, goal=2048):
        super(Game2048Env, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0,
            high=goal,
            shape=(self.size, self.size),
            dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy(), {}

    def step(self, action):
        '''
        0: Up
        1: Down
        2: Left
        3: Right
        4: Up Left
        5: Up Right
        6: Down Left
        7: Down Right
        '''
        moved = False
        score = 0
        if action == 0:
            self.board = np.rot90(self.board, 1)
            moved, score = self.move()
            self.board = np.rot90(self.board, -1)

        elif action == 1:
            self.board = np.rot90(self.board, -1)
            moved, score = self.move()
            self.board = np.rot90(self.board, 1)

        elif action == 2:
            moved, score = self.move()

        elif action == 3: 
            self.board = np.fliplr(self.board)
            moved, score = self.move()
            self.board = np.fliplr(self.board)
        
        elif action == 4:
            self.board = rot45(self.board, 1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, -1)

        elif action == 5:
            self.board = np.fliplr(self.board)
            self.board = rot45(self.board, 1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, -1)
            self.board = np.fliplr(self.board)

        elif action == 6:
            self.board = rot45(self.board, -1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, 1)

        elif action == 7:
            self.board = np.fliplr(self.board)
            self.board = rot45(self.board, -1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, 1)
            self.board = np.fliplr(self.board)

        else:
            raise ValueError("Invalid action")

        if moved:
            self.add_random_tile()

        done = not self.can_move()
        reward = score
        info = {}
        return self.board.copy(), reward, done, False, info
    
    # FOR TESTING ONLY
    def test_step(self, action):
        '''
        0: Up
        1: Down
        2: Left
        3: Right
        4: Up Left
        5: Up Right
        6: Down Left
        7: Down Right
        '''
        moved = False
        score = 0
        if action == 0:
            self.board = np.rot90(self.board, 1)
            moved, score = self.move()
            self.board = np.rot90(self.board, -1)

        elif action == 1:
            self.board = np.rot90(self.board, -1)
            moved, score = self.move()
            self.board = np.rot90(self.board, 1)

        elif action == 2:
            moved, score = self.move()

        elif action == 3: 
            self.board = np.fliplr(self.board)
            moved, score = self.move()
            self.board = np.fliplr(self.board)
        
        elif action == 4:
            self.board = rot45(self.board, 1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, -1)

        elif action == 5:
            self.board = np.fliplr(self.board)
            self.board = rot45(self.board, 1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, -1)
            self.board = np.fliplr(self.board)

        elif action == 6:
            self.board = rot45(self.board, -1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, 1)

        elif action == 7:
            self.board = np.fliplr(self.board)
            self.board = rot45(self.board, -1)
            moved, score = self.move(diagonal=True)
            self.board = rot45(self.board, 1)
            self.board = np.fliplr(self.board)

        else:
            raise ValueError("Invalid action")

    def render(self):
        print(self.board)

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[row][col] = 2 if np.random.random() < 0.9 else 4

    def can_move(self):
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
        tl_br_diag_matrix = rot45(self.board, 1)
        for row in tl_br_diag_matrix:
            for col in range(len(row) - 1):
                if row[col] == row[col+1]:
                    return True
        tr_bl_diag_matrix = rot45(self.board, -1)
        for row in tr_bl_diag_matrix:
            for col in range(len(row) - 1):
                if row[col] == row[col+1]:
                    return True

        return False

    def move(self, diagonal=False):
        if not diagonal:
            return self.move_square()
        else:
            return self.move_rotated()
    
    def move_square(self):
        #print('Square')
        moved = False
        score = 0
        
        for row in range(self.size):
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
                    merged_tiles.append(tiles[i] * 2)
                    score += tiles[i] * 2
                    skip = True
                else:
                    merged_tiles.append(tiles[i])
                i += 1
            merged_tiles += [0] * (self.size - len(merged_tiles))
            if not np.array_equal(self.board[row], merged_tiles):
                moved = True
            self.board[row] = merged_tiles
        return moved, score

    def move_rotated(self):
        #print("Rotated")
        moved = False
        score = 0
        
        for row in range(len(self.board)):
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
                    merged_tiles.append(tiles[i] * 2)
                    score += tiles[i] * 2
                    skip = True
                else:
                    merged_tiles.append(tiles[i])
                i += 1
            merged_tiles += [0] * (self.board[row].shape[0] - len(merged_tiles))
            if not np.array_equal(self.board[row], merged_tiles):
                moved = True
            self.board[row] = merged_tiles
        return moved, score
    
if __name__ == "__main__":

    env = Game2048Env()
    observation, info = env.reset()
    done = False
    total_score = 0

    matrix = np.zeros((4, 4))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = (i)*(matrix.shape[1]) + (j+1)

    env.board = matrix
    env.render()
    print("Initial state\n")

    env.board = rot45(env.board, 1)
    env.render()
    env.move(diagonal=True)
    env.board = rot45(env.board, -1)

    env.board = rot45(env.board, -1)
    env.board = rot45(env.board, -1)
    env.render()
