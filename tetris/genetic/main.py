
# tetris game and AI
from tetris_game import TetrisApp
from tetris_game_half import TetrisApp as TetrisAppHalf
from tetris_ai import TetrisAI
import threading


# Half Tetris
app = TetrisAppHalf()
ai = TetrisAI(app)

threading.Thread(target=app.run).start()
ai.start(num_units=200, max_gen=20,elitism_rate=0.4, crossover_rate=0.4, mutation_val=0.1, target_file="tetrishalf.csv")

"""
# Tetris Limited stones
app = TetrisApp()
ai = TetrisAI(app)

threading.Thread(target=app.run).start()
ai.start(max_stones=1000, num_units=200, max_gen=20,elitism_rate=0.4, crossover_rate=0.4, mutation_val=0.1, target_file="tetrislim.csv")
"""





# Testers
#ai.start(max_stones=200, seed=(-0.525, -0.284, -0.685, 0.873))
#ai.start(max_stones=200, num_units=20, max_gen=10,elitism_rate=0.3, crossover_rate=0.2, mutation_val=0.3)



