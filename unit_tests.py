import numpy as np
from game_env import Game2048Env

def runActionTest(env, initial_board, action_sequence, expected_board):
  env.board = initial_board
    
  for n in action_sequence:
    env.test_step(int(n))

  return np.array_equal(env.board, expected_board), env.board, expected_board

def runRotateTest(env, initial_board, action_sequence, expected_board):
  env.board = initial_board

  for n in action_sequence:
    env.board = env.rot45(env.board, 1 if int(n) else -1)

  if isinstance(expected_board, list) and isinstance(expected_board, list):
    if len(expected_board) != len(env.board):
      return False, env.board, expected_board
    passed = True
    for i, row in enumerate(expected_board):
      passed = passed and np.array_equal(row, env.board[i])
    return passed, env.board, expected_board
    
  else:
    passed = np.array_equal(env.board, expected_board)
    return passed, env.board, expected_board

def runCanMoveTest(env, initial_board, expected):
  env.board = initial_board

  return env.can_move() == expected, initial_board, expected

def runAllActionTests():
  print("RUNNING ACTION TESTS")
  env = Game2048Env()

  #Up left
  case1 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     "4",
     np.array([[2,2,2,1], 
               [2,2,1,0], 
               [2,1,0,0],
               [1,0,0,0]], dtype=np.int64)
  )
  #2 x Up left
  case2 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     "44",
     np.array([[4,2,2,1], 
               [2,0,1,0], 
               [2,1,0,0],
               [1,0,0,0]], dtype=np.int64)
  )
  #2 x Up left, Up right
  case3 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     "445",
     np.array([[4,4,4,2], 
               [0,0,2,0], 
               [0,0,0,0],
               [0,0,0,0]], dtype=np.int64)
  )
  #2 x Up left, Up right, Down right
  case4 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     "4457",
     np.array([[0,0,0,2], 
               [0,0,4,4], 
               [0,0,0,2],
               [0,0,0,4]], dtype=np.int64)
  )
  #2 x Up left, Up right, Down right, Down left
  case5 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     "44576",
     np.array([[0,0,0,0], 
               [0,0,0,0], 
               [0,2,0,0],
               [4,4,2,4]], dtype=np.int64)
  )

  test_cases = [case1, case2, case3, case4, case5]
  i = 0
  for initial_board, action_sequence, expected_board in test_cases:
    passed, acc_board, exp_board = runActionTest(env, initial_board, action_sequence, expected_board)
    
    if not passed:
       print(f"Failed test {i}")
       print(acc_board)
       print(exp_board)

    else:
       print(f"Passed test {i}")
    
    i+=1

def runAllRotateTests():
  print("RUNNING ROTATE TESTS")
  env = Game2048Env()

  #2 cw 2 ccw
  case1 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     "0011",
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64)
  )
  #1 ccw 1 cw
  case2 = (
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64),
     "10",
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64)
  )
  #1 cw 1 ccw
  case3 = (
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64),
     "01",
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64)
  )
  #2 ccw
  case4 = (
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64),
     "11",
     np.array([[4,8,12,16], 
               [3,7,11,15], 
               [2,6,10,14], 
               [1,5,9,13]], dtype=np.int64)
  )
  #2 cw
  case5 = (
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64),
     "00",
     np.array([[13,9,5,1], 
               [14,10,6,2], 
               [15,11,7,3], 
               [16,12,8,4]], dtype=np.int64)
  )
  #1 ccw 
  case6 = (
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64),
     "1",
     [np.array([4], dtype=np.int64), 
      np.array([3,8], dtype=np.int64), 
      np.array([2, 7, 12], dtype=np.int64), 
      np.array([1, 6, 11, 16], dtype=np.int64),
      np.array([5, 10, 15], dtype=np.int64),
      np.array([9, 14], dtype=np.int64),
      np.array([13], dtype=np.int64)]
  )
  #1 cw 
  case7 = (
     np.array([[1,2,3,4], 
               [5,6,7,8], 
               [9,10,11,12], 
               [13,14,15,16]], dtype=np.int64),
     "0",
     [np.array([1], dtype=np.int64), 
      np.array([5,2], dtype=np.int64), 
      np.array([9, 6, 3], dtype=np.int64), 
      np.array([13, 10, 7, 4], dtype=np.int64),
      np.array([14, 11, 8], dtype=np.int64),
      np.array([15, 12], dtype=np.int64),
      np.array([16], dtype=np.int64)]
  )

  test_cases = [case1, case2, case3, case4, case5, case6, case7]
  i = 0
  for initial_board, action_sequence, expected_board in test_cases:
    passed, acc_board, exp_board = runRotateTest(env, initial_board, action_sequence, expected_board)
    
    if not passed:
       print(f"Failed test {i}")
       print(acc_board)
       print(exp_board)

    else:
       print(f"Passed test {i}")
    
    i+=1

def runAllCanMoveTests():
  print("RUNNING CAN MOVE TESTS")
  env = Game2048Env()

  #Possible in all directions
  case1 = (
     np.array([[1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1], 
               [1,1,1,1]], dtype=np.int64),
     True
  )
  #Possible only horizontal
  case2 = (
     np.array([[1,1,1,1], 
               [2,2,2,2], 
               [1,1,1,1], 
               [2,2,2,2]], dtype=np.int64),
     True
  )
  #Possible only vertical
  case3 = (
     np.array([[2,1,2,1], 
               [2,1,2,1], 
               [2,1,2,1], 
               [2,1,2,1]], dtype=np.int64),
     True
  )
  #Possible only tl_br diagonal
  case4 = (
     np.array([[4,2,1,4], 
               [1,8,16,2], 
               [2,32,8,1], 
               [4,1,2,4]], dtype=np.int64),
     True
  )
  #Possible only tr_bl diagonal
  case5 = (
     np.array([[4,2,1,4], 
               [1,32,8,2], 
               [2,8,16,1], 
               [4,1,2,4]], dtype=np.int64),
     True
  )
  #Not possible
  case6 = (
     np.array([[4,2,1,4], 
               [1,8,16,2], 
               [2,64,32,1], 
               [4,1,2,4]], dtype=np.int64),
     False
  )

  test_cases = [case1, case2, case3, case4, case5, case6]
  i = 0
  for initial_board, expected in test_cases:
    passed, initial_board, expected = runCanMoveTest(env, initial_board, expected)
    
    if not passed:
       print(f"Failed test {i}")
       print(initial_board)
       print(expected)

    else:
       print(f"Passed test {i}")
    
    i+=1

if __name__ == '__main__':
  runAllActionTests()
  runAllRotateTests()
  runAllCanMoveTests()

  
        



