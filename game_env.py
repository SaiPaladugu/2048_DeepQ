import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=4, goal=2048):
        super(Game2048Env, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)
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
        '''
        moved = False
        score = 0
        if action == 0:
            self.board = np.rot90(self.board, -1)
            moved, score = self.move()
            self.board = np.rot90(self.board)

        elif action == 1:
            self.board = np.rot90(self.board, 1)
            moved, score = self.move()
            self.board = np.rot90(self.board, -1)

        elif action == 2:
            moved, score = self.move()

        elif action == 3: 
            self.board = np.fliplr(self.board)
            moved, score = self.move()
            self.board = np.fliplr(self.board)

        else:
            raise ValueError("Invalid action")

        if moved:
            self.add_random_tile()

        done = not self.can_move()
        reward = score
        info = {}
        return self.board.copy(), reward, done, False, info

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
        return False

    def move(self):
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
