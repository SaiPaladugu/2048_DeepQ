import gymnasium as gym
from gymnasium import spaces
import numpy as np

# reward strucutre:
# -1 per step
# +2048 per game win
# +k * log_2(max_tile)

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(
            low=0,
            high=2048,
            shape=(self.size, self.size),
            dtype=np.int32
        )
        self.reset()
        self.global_max_tile = 0
        self.k = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.global_max_tile = max(self.add_random_tile(), self.add_random_tile())
        
        return self.board.copy(), {}

    def step(self, action):
        # Rotate the board and merge based on the action
        if action == 0:  # up
            self.board = np.rot90(self.board, -1)
            moved, score, max_tile = self.move()
            self.board = np.rot90(self.board)
        elif action == 1:  # down
            self.board = np.rot90(self.board, 1)
            moved, score, max_tile = self.move()
            self.board = np.rot90(self.board, -1)
        elif action == 2:  # left
            moved, score, max_tile = self.move()
        elif action == 3:  # right
            self.board = np.fliplr(self.board)
            moved, score, max_tile = self.move()
            self.board = np.fliplr(self.board)
        else:
            raise ValueError("Invalid action")

        if moved:
            self.add_random_tile()
        
        done = not self.can_move() or max_tile == 2048
        reward = -1
        if self.global_max_tile < max_tile:
            self.global_max_tile = max_tile
            reward += self.k * np.log(self.global_max_tile) 
        
        return self.board, score, done, False, {}, reward

    def render(self):
        print(self.board)

    def add_random_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if empty_cells.size > 0:
            row, col = empty_cells[np.random.randint(len(empty_cells))]
            rand_float = np.random.random()
            self.board[row, col] = 2 if rand_float < 0.9 else 4
            return self.board[row, col]

    def can_move(self):
        if np.any(self.board == 0):
            return True
        return np.any(self.board[:, :-1] == self.board[:, 1:]) or np.any(self.board[:-1, :] == self.board[1:, :])

    def move(self):
        moved = False
        score, max_tile = 0, 0
        for row in range(self.size):
            tiles = self.board[row][self.board[row] != 0]  # get non-zero tiles
            if len(tiles) == 0:
                continue
            merged = []
            skip = False
            for i in range(len(tiles)):
                tileToAppend = tiles[i]
                if skip:
                    skip = False
                    continue
                if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                    tileToAppend =  tiles[i] * 2
                    score += tileToAppend
                    skip = True
                merged.append(tileToAppend)
                max_tile = max(max_tile, tileToAppend)
            merged.extend([0] * (self.size - len(merged)))
            if not np.array_equal(self.board[row], merged):
                moved = True
            self.board[row] = merged
        return moved, score, max_tile