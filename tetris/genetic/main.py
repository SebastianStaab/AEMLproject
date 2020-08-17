
# tetris game and AI
#from tetris import TetrisApp
from tetris_game_half import TetrisApp
from tetris_ai import TetrisAI
from multiprocessing import Process

import time, threading

"""
def tetris_p():
    app = TetrisApp()
    ai = TetrisAI(app)

    threading.Thread(target=app.run).start()


if __name__ == '__main__':

 

  uncomment to run multiple trainings at one time

  processes = []
  num_tetris_ai = 5

  for m in range(num_tetris_ai):
    p = Process(target=tetris_p)
    p.start()
    processes.append(p)

  for p in processes:
    p.join()
"""
  
app = TetrisApp()
ai = TetrisAI(app)

threading.Thread(target=app.run).start()

ai.start(seed=(-0.525, -0.284, -0.685, 0.873))
#ai.start(20)



