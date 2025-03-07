from Game import Game
if __name__ == '__main__':
    g = Game()
    g.print_state()
    while not g.is_done():
        action = [int(x) for x in list(input().split())]
        g.perform_action(action[0], action[1], action[2])
        g.print_state()
    g.print_blocks()