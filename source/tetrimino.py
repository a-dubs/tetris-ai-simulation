import random


class Tetrimino:
    shapes = {
        'O':
        [['y', 'y'],
         ['y', 'y']],
        'I':
        [[' ', ' ', ' ', ' '],
         [' ', ' ', ' ', ' '],
         ['c', 'c', 'c', 'c'],
         [' ', ' ', ' ', ' ']],
        'T':
        [[' ', ' ', ' '],
         ['p', 'p', 'p'],
         [' ', 'p', ' ']],
        'L':
        [[' ', ' ', ' '],
         ['o', 'o', 'o'],
         [' ', ' ', 'o']],
        'J':
        [[' ', ' ', ' '],
         ['b', 'b', 'b'],
         ['b', ' ', ' ']],
        'S':
        [[' ', ' ', ' '],
         ['g', 'g', ' '],
         [' ', 'g', 'g']],
        'Z':
        [[' ', ' ', ' '],
         [' ', 'r', 'r'],
         ['r', 'r', ' ']],
    }

   

    def __init__(self, shape=None, x=None, y=None, orientation='N', ldml=15):
        # TODO: if shape == None
        # TODO: if xpos == None
        # TODO: if ypos == None
        self.minos = self.get_minos(self.rand_shape() if shape == None else shape)
        self.ldml = ldml  # lock down movements left - number of moves left after landing before locking down
        self.landed = False
        self.landed_y = -1
        self.ld_timer = -1
        self.lowest_y = y
        # x coordinate of bottom left corner of tetrimino - D: [1, playfield width]
        self.x = x
        # y coordinate of bottom left corner of tetrimino - D: [1, playfield height]
        self.y = y
        # one of ('N', 'E', 'S', W') reprenting the current tetrimino's orientation
        self.orientation = orientation
        self.size = len(self.minos)

    @staticmethod
    def get_minos(shape):
        return Tetrimino.shapes[shape.upper()]

    # returns a random shape represented by its respective characters #
    @staticmethod
    def rand_shape():
        return random.choice(list(Tetrimino.shapes.keys()))

    @staticmethod
    def make_bag(repetitions):
        bag =  [list(Tetrimino.shapes.keys())[ i % len(Tetrimino.shapes.keys()) ] 
            for i in range(0, len(Tetrimino.shapes.keys()) * repetitions)]
        random.shuffle(bag)
        return bag

    def reset_LDML(self):
        self.ldml = 15

    def log(self):
        print("Facing " + self.orientation)
        print("Coords: " + self.coords())
        print("----")
        for y in range(-1, -len(self.minos)-1, -1):
            print("".join(self.minos[y]))
        print("----")
    
    def coords(self):
        return "("+str(self.x)+","+str(self.y)+")"

# Tetrimino('T', 1 , 1).log()
