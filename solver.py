from datetime import datetime
from Game import Game
if __name__ == '__main__':
    g = Game()
    g.print_state()
    g._get_best_turn()
    print(datetime.now())
