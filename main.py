from Game import Game
if __name__ == '__main__':
    g = Game()
    # print(g.blocks)
    while not g.is_done():
        g._get_next_3()
        action = [int(x) for x in list(input().split())]
        g.perform_action(action[0], action[1], action[2])
        g.print_matrix()
    g.print_blocks()