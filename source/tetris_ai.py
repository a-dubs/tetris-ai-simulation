# #==================================================================================================================# #
# #==================================================================================================================# #
# #==================================================== IMPORTS =====================================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #
from tetrimino import Tetrimino
import random
import numpy as np
from copy import copy, deepcopy
from time import time

# #==================================================================================================================# #
# #==================================================================================================================# #
# #==================================================== HELPER FUNCTIONS =====================================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #


# #==================================================================================================================# #
# #==================================================================================================================# #
# #==================================================== TETRIS AI CLASS =====================================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #


class TetrisAI:

    param_labels = [
        "cliff_l_e",
        "cliff_h_e",
        "cliff_l_w",
        "cliff_h_w",
        "m_stack_w",
        "m_stack_e",
        "a_stack_w",
        "a_stack_e",
        "stack_d_w",
        "stack_d_e",
        "stack_d_thresh",
        "score_w",
        "go_w"
    ]
        
    param_weight_bounds = {
        "cliff_l_e" : ( 0.0 , 5.0 ),
        "cliff_h_e" : ( 0.0 , 5.0 ),
        "cliff_l_w" : ( -10.0 , 0.0 ),
        "cliff_h_w" : ( -10.0 , 0.0 ),
        "m_stack_w" : ( -10.0 , 0.0 ),
        "m_stack_e" : ( 0.0 , 5.0 ),
        "a_stack_w" : ( -10.0 , 0.0 ),
        "a_stack_e" : ( 0.0 , 5.0 ),
        "stack_d_w" : ( -10.0 , 0.0 ),
        "stack_d_e" : ( 0.0 , 5.0 ),
        "stack_d_thresh": (0, 15),
        "score_w": (0.0, 10.0),
        "go_w" : ( -10.0 , 0.0 ),
    }

    moves = ["fall", "left", "right", "ccw", "cw"]  # all possible moves
    
    def __init__(self, playfield, tetrimino, next_tet, mps, level, method="greedy", param_weights = [], params = {}, debug = False):
        self.tet = tetrimino  # Tetrimino In Play
        self.pf = playfield
        self.pf_w = len(playfield[0])
        self.pf_h = len(playfield)
        self.next_tet = next_tet
        self.mps = mps
        self.lvl = level
        self.method = method
        self.timer = 0
        self.debug = debug
        
        if not param_weights:
            param_weights = [round(random.uniform(*TetrisAI.param_weight_bounds[param]), 2) for param in TetrisAI.param_labels]
        
        if not params:
            self.params = {
                self.param_labels[i] : param_weights[i] for i in range(len(self.param_labels))
            }
        self.params = params
    
    @staticmethod   
    def game_over(pf):
        return sum([sum([1 if mino != ' ' else 0 for mino in pf[y]]) for y in range(len(pf)) if y >= len(pf)-4]) > 0

    @staticmethod
    def score_move(clears, level):
        clear_rewards = [0, 100, 300, 500, 800]
        return clear_rewards[clears] * level

    @staticmethod
    def in_bounds(pf, x, y):
        skybox_height = 4
        """print(r:= x >= 1 and x <= len(pf[0]) and y >= 1 and y <= len(pf))
        print(x)
        print(y)"""
        return  x >= 1 and x <= len(pf[0]) and y >= 1 and y <= len(pf)

    @staticmethod
    def get_clears(pf):
        return sum([1 if row.count(' ') == 0 else 0 for row in pf])

    # clears any rows that need cleared and returns number of lines cleared #
    @staticmethod
    def make_clears(pf):
        clears = 0
        for i in range(-1, -len(pf)-1, -1):
            if pf[i].count(' ') == 0:
                del pf[i]
                pf.append([' ' for x in range(len(pf[0]))])
                clears+=1
        return clears

    @staticmethod
    def place_tetrimino(pf, tet):
        for row in range(tet.size):
            for col in range(tet.size):
                if tet.minos[row][col] != ' ':
                    try:
                        pf[tet.y + row - 1][tet.x + col - 1] = tet.minos[row][col]
                    except:
                        print("------------------------")
                        print(tet.x)
                        print(tet.y)
                        print(tet.minos[row][col])
                        print(pf[tet.y + row - 1][tet.x + col - 1])

        return pf

    # mutates given playfield: places tetrimino & make any necessary row clears. Return the number of rows cleared #
    @staticmethod
    def update_playfield(pf, tet):
        TetrisAI.place_tetrimino(pf, tet)
        return TetrisAI.make_clears(pf)
    
    # TODO: add heuristic eval that minimizes the number of overhangs created (mino with empty cell below) 
        # maybe add multiplier for length of overhang? ie overhang of 2 is worse than 1
    # TODO: maybe also add even bigger penatly for holes (empty cell surrounded by minos)

    def greedy_heuristic_evaluation(self, tet, pf = None):
        if not pf:
            pf = self.pf
        pf = deepcopy(pf)
        pfw = len(pf[0])
        pfh = len(pf)
        

        clears = TetrisAI.update_playfield(pf, tet)
        score = TetrisAI.score_move(clears, 1)
        max_stack_height = sum([int(row.count(' ') < pfw) for row in pf])
        cliffs = [[int(pf[row][col] != ' ' and pf[row - 1][col] == ' ') for col in range(pfw)] for row in range(1, pfh - 4)]

        cliff_heights = [[0 for item in row] for row in cliffs]
        for col in range(pfw):
            prev_row = -1
            for row in range(len(cliffs)-1, -1, -1):
                if cliffs[row][col] == 1 and prev_row == -1:
                    prev_row = row
                    #print("cliff reached")
                 
                elif pf[row][col] != ' ' and prev_row != -1:
                    cliff_heights[prev_row][col] = ((prev_row - row)**self.params["cliff_h_e"]) * self.params["cliff_h_w"]
                    #print("edge reached")
                    #print(row)
                    #print(prev_row)
                    
                # catch bottommost edge case
                if row == 0 and prev_row != -1 and pf[row][col] == ' ':
                    cliff_heights[prev_row][col] = ((prev_row - row + 1) ** self.params["cliff_h_e"]) * self.params["cliff_h_w"]

        cliff_heights = sum([sum(row) for row in cliff_heights])
        
        stack_danger = sum([int(pf[row].count(' ') < pfw) * ((row - self.params["stack_d_thresh"]) ** self.params["stack_d_e"]) for row in range(int(self.params["stack_d_thresh"])+1, len(pf))])
        

        cliff_lengths = 0
        for row in range(len(cliffs)):
            prev_i = -1
            for i in range(pfw):
                if cliffs[row][i] > 0 and prev_i == -1:
                    prev_i = i
                 
                elif cliffs[row][i] == 0 and prev_i != -1:
                    cliff_lengths += (i - prev_i) ** self.params["cliff_l_e"] * self.params["cliff_l_w"]
                    
                # catch rightmost edge case
                if i == pfw - 1 and prev_i != -1 and cliffs[row][i] > 0:
                    cliff_lengths += (i - prev_i + 1) ** self.params["cliff_l_e"] * self.params["cliff_l_w"]
                    
        a_stacks = sum([max([row * int(pf[row][col] != ' ') for row in range(len(pf))]) for col in range(pfw)])/(1.0 * pfw)
        
        
        eval = 0
        eval += max_stack_height ** self.params["m_stack_e"] * self.params["m_stack_w"]
        eval += a_stacks ** self.params["a_stack_e"] * self.params["a_stack_w"]
        eval += stack_danger * self.params["stack_d_w"]
        eval += cliff_lengths
        eval += cliff_heights
        eval += clears * self.params["score_w"]
        eval += int(self.game_over(pf)) * self.params["go_w"]
        #print(eval)
        return eval
 

    @staticmethod
    def valid_location(pf, tet):

        for row in range(tet.size):
            for col in range(tet.size):
                if ((ib := TetrisAI.in_bounds(pf, tet.x + col, tet.y + row)) and tet.minos[row][col] != ' '
                and pf[tet.y + row - 1][tet.x + col - 1] != ' ') or (not ib and tet.minos[row][col] != ' '):
                    #print("tet (x, y): (" + str(tet.x) + ", " + str(tet.y) + ")")
                    #print("minos (c, r): (" + str(col) + ", " + str(row) + ")")
                    #print("mino (x, y): (" + str(tet.x + col) + ", " + str(tet.y + row) + ")")
                    #print(tet.minos[row][col])
                    return False

        return True

    # rotates given Tetrimino object and adjusts all its necessary valffffffffues. returns true if success, otherwise returns false
    @staticmethod
    def rotate_tetrimino(pf, tet, dir):
        mult = None
        if dir in ("ccw", "left", "l"):
            mult = -1
        elif dir in ("cw", "right", "r"):
            mult = 1
        else:
            return False
        t = np.array(tet.minos)
        minos = []
        for i in range(len(t[0])):
            minos.append(t[:, -i-1].tolist() if mult == 1 else t[::-1,i].tolist())
                
        tet.minos = minos
        dirs = ('N', 'E', 'S', 'W')
        tet.orientation = dirs[(dirs.index(tet.orientation) + mult) % 4]
        #Tetrimino.log(tet)

        # push / force tetrimino --> <-- so that it is not overflowing the playfield
        xshift = 0
        if tet.x < 1:
            for col in range(tet.size):
                for row in range(tet.size):
                    if xshift == 0 and tet.minos[row][col] != ' ':
                        xshift = 1 - tet.x + col
                        break

        elif tet.x > len(pf[0]) + 1 - tet.size:
            for col in range(-1, -tet.size - 1, -1):
                for row in range(tet.size):
                    if xshift == 0 and tet.minos[row][col] != ' ':
                        xshift = len(pf[0]) - tet.x - tet.size - col 
                        break
        tet.x += xshift
        while not TetrisAI.valid_location(pf, tet) and tet.y < 22:
            tet.y += 1

        if tet.landed:
            tet.ldml -= 1
            tet.ld_timer = time()
        #print("ldml: " + str(tet.ldml))
        #print("y val: " + str(tet.y))
        return True

    @staticmethod
    def execute_move(pf, tet, move):
        if move == "left" or move =="right":
            TetrisAI.move_tetrimino(pf, tet, move)
        elif move == "cw" or move == "ccw":
            TetrisAI.rotate_tetrimino(pf, tet, move)
        elif move == "drop":
            TetrisAI.drop_tetrimino(pf, tet)

    @staticmethod
    def move_tetrimino(pf, tet, dir):
        
        if dir in ("left", "l", "west", "w"):
            xshift = - 1
        elif dir in ("right", "r", "east", "e"):
            xshift = 1    
        else:
            return False
        
        before = tet.x
        tet.x += xshift
        if not TetrisAI.valid_location(pf, tet):
            tet.x -= xshift
        elif tet.landed:
            tet.ldml -= 1
            tet.ld_timer = time()
        #print(tet.ldml)
        #print("move_tetrimino called! before: " + str(before) + " -> " + str(tet.x))
        return tet.x != before 

    # returns true if the tetrimino was able to be dropped # 
    @staticmethod
    def drop_tetrimino(pf, tet):
        start = tet.y
        while TetrisAI.valid_location(pf, tet):
            tet.y -= 1
        tet.y+=1
        tet.y = start if tet.y > start else tet.y
        tet.ldml = 0
        return tet.y != start

    @staticmethod
    def can_fall(pf, tet):
        t = deepcopy(tet)
        t.y -= 1
        return TetrisAI.valid_location(pf, t)

    # takes in the active tetrimino, last time when gravity occured, the current time, and the current level #
    # returns true if gravity was applied on this call. updates prev_time to now if gravity was applied #  
    @staticmethod
    def gravity(pf, tet, prev, now, level):
        interval = (0.8 - ((level - 1) * 0.0007))**(level - 1)
        if now >= prev + interval and TetrisAI.can_fall(pf, tet):
            tet.y -= 1
            if tet.y < tet.lowest_y:
                tet.lowest_y = tet.y
                tet.reset_LDML()
            #print("GRAVITYYYYYY")
            return now
        return prev

    @staticmethod
    def log_pf(pf, tet):
        p = deepcopy(pf)
        TetrisAI.place_tetrimino(p, tet)
        for i in range(len(p)):

            print("".join([mino if mino!= ' ' else 'X' for mino in p[-i-1]]))

    @staticmethod
    def tetrimino_has_landed(pf, tet):
        tet.landed = not TetrisAI.can_fall(pf, tet)
        return tet.landed

    @staticmethod
    def tetrimino_locked_down(pf, tet, now):
        return tet.ldml == 0 or (now - 0.5 >= tet.ld_timer and not TetrisAI.can_fall(pf, tet))

    # update the AI's playfield info. can specifiy new level value too. #
    def update(self, playfield, level=None):
        self.playfield = playfield
        self.level = level

    def interest_left(self, state, pf = None):
        if not pf:
            pf = self.pf
        t = state["tetrimino"]
        leftmost = self.tet_leftmost_mino_pos(t)
        lx = leftmost[0] - 1
        ly = leftmost[1] - 1
        
        if lx > 0:
            return self.pf[ly][lx-1] == ' ' and self.pf[ly+1][lx-1] != ' '
        else:
            return False

    def interest_right(self, state, pf = None):
        if not pf:
            pf = self.pf
        t = state["tetrimino"]
        rightmost = self.tet_rightmost_mino_pos(t)
        rx = rightmost[0] - 1
        ry = rightmost[1] - 1
        if rx + 1 < self.pf_w:
            return pf[ry][rx+1] == ' ' and pf[ry+1][rx+1] != ' '
        else:
            return False
   
    @staticmethod
    def tet_leftmost_mino_pos(tet):
        for row in range(tet.size):
            for col in range(tet.size):
                if tet.minos[row][col] != ' ':
                    return (tet.x + col, tet.y + row)

    @staticmethod
    def tet_rightmost_mino_pos(tet):
        for row in range(tet.size):
            for col in range(-1, -tet.size - 1, -1):
                if tet.minos[row][col] != ' ':
                    return (tet.x + tet.size + col, tet.y + row)

    def valid_move(self, state, move, pf=None):
        if not pf:
            pf = self.pf
        if move == "left" or move == "right":
            return self.move_tetrimino(pf, deepcopy(state["tetrimino"]), move)

        elif move == "ccw" or move == "cw":
            return self.rotate_tetrimino(pf, deepcopy(state["tetrimino"]), move)

        elif move == "fall":
            return TetrisAI.can_fall(pf, state["tetrimino"])
        elif move == "":
            pass
        else:
            print("INVALID MOVE GIVEN TO valid_move(): " + str(move))
            return False
    
    # filters list of possible moves to be passed on to next decision node #
    def filter_child_moves(self, tet, move, parent_moves, pf=None):
        if not pf:
            pf = self.pf
        child_moves = copy(parent_moves)
        if move == "left":
            if "right" in parent_moves:
                child_moves.remove("right")

        elif move == "right":
            if "left" in parent_moves:
                child_moves.remove("left") 

        elif move == "ccw":
            if "right" in parent_moves:
                child_moves.remove("right")
            if "left" in parent_moves:
                child_moves.remove("left")
            if "cw" in parent_moves:
                child_moves.remove("cw")
            if tet.orientation == 'E':
                child_moves.remove("ccw")
        
        elif move == "cw":
            if "right" in parent_moves:
                child_moves.remove("right")
            if "left" in parent_moves:
                child_moves.remove("left")
            if "ccw" in parent_moves:
                child_moves.remove("ccw")
            if tet.orientation == 'W':
                child_moves.remove("cw")

        elif move == "fall":
            child_moves = ["fall"]
        elif move == "":
            pass
        else:
            print("INVALID MOVE GIVEN TO filter_child_moves()")
        
        if "fall" in parent_moves and self.tetrimino_has_landed(pf, tet):
            child_moves.remove("fall")
            
       

        return child_moves

    def make_child_state(self, state, move, pf=None):
        if not pf:
            pf = self.pf
        s = deepcopy(state)
        
        if move == "left" or move =="right":
            self.move_tetrimino(pf, s["tetrimino"], move)
            
        elif move == "cw" or move == "ccw":
            prev_min_x = self.tet_leftmost_mino_pos(s["tetrimino"])[0]
            prev_max_x = self.tet_rightmost_mino_pos(s["tetrimino"])[0]
            self.rotate_tetrimino(pf, s["tetrimino"], move)
            #print("child tet orientation: " + str( s["tetrimino"].orientation))
        else:
            pass
        
        s["timer"] += 1.0 / self.mps
        s["gravity_timer"] = self.gravity(pf, s["tetrimino"], s["gravity_timer"], s["timer"], self.lvl)
        s["parent_move"] = move
        s["inherited_moves"] = self.filter_child_moves(s["tetrimino"], move, s["inherited_moves"])
        if move == "ccw" or move == "cw":
            #print("prev_max_x: " + str(prev_max_x))
            #print("rightmost: " + str(self.tet_rightmost_mino_pos(s["tetrimino"])))
            if prev_max_x == self.pf_w and self.tet_rightmost_mino_pos(s["tetrimino"])[0] < prev_max_x:
                #print("yeet_r")
                s["inherited_moves"].append("right")
                if move in s["inherited_moves"]:
                    s["inherited_moves"].remove(move)
            elif prev_min_x == 1 and self.tet_leftmost_mino_pos(s["tetrimino"])[0]  > prev_min_x:
                s["inherited_moves"].append("left")
                if move in s["inherited_moves"]:
                    s["inherited_moves"].remove(move)
                #print("yeet_l")
        return s 
        
    def get_child_states(self, state, pf=None):
        if not pf:
            pf = self.pf
        child_states = []
        parent_moves = []
        parent_moves += state["inherited_moves"]
        if not "left" in parent_moves and self.interest_left(state, pf):
            #state["tetrimino"].log()
            #print("INTEREST LEFT!")
            parent_moves.append("left")
        if not "right" in parent_moves and self.interest_right(state, pf):
            #print("INTEREST RIGHT!")
            parent_moves.append("right")
        for move in parent_moves:
            if self.valid_move(state, move):
                child_states.append(self.make_child_state(state, move, pf))
        return child_states

    @staticmethod
    def get_simplified_path(path):
        i = 0
        for i in range(-2, -len(path) - 1, -1):
            if path[i] != "fall":
                break
            elif i == -len(path):
                i -= 1
                break
        return path[:i+1] + ["drop"]
    
    def get_best_moves(self):
        
        state = {
            "tetrimino": deepcopy(self.tet),
            "parent_move": "", 
            "inherited_moves": self.moves,  # all remaining moves this state can do based on its ancestors' moves
            "timer": 0,
            "gravity_timer": 0 
        }

        # get valid moves for start state
        
        # calculate list containing the modified state for each valid move

        # call max_agent(state) on each new state

        # increment timer

        # return the move that results in the max utility out of the utilities given by max_agent(stete)
        #self.log_pf(self.pf, state["tetrimino"])
        start = time()

        if self.method == "greedy":
            result = self.max_node(state, "path")

        elif self.method == "yield_to_next":
            temp_state = deepcopy(state)
            temp_state["tetrimino"] = self.next_tet
            next_tet_opt_loc = self.max_node(temp_state, "tetrimino")[1]
            temp_pf = deepcopy(self.pf)

            self.place_tetrimino(temp_pf, next_tet_opt_loc)
            result = self.max_node(state, return_type="path", pf=temp_pf)
            
        else:
            print("you messed up! invalid return type given to get_best_moves()!")

        #print("optimal answer found in " + str(round(time() - start, 4)) + " seconds!")
        #print(result)
        return result

    def max_node(self, state, return_type = "path", pf = None):
        child_states = self.get_child_states(state, pf)
        #self.log_pf(self.pf, state["tetrimino"])
        
        if not pf:
            pf = self.pf
        if self.debug and (len(child_states) != 1 or child_states[0]["parent_move"] != "fall"):
            #print("time: " + str(state["timer"]))
            #print("parent move: " + state["parent_move"])
            #print("child state moves: " + " ".join([cs["parent_move"] for cs in child_states]))
            #print("tet location: " + state["tetrimino"].coords())
            #print("tet orientation: " + state["tetrimino"].orientation)
            #print("tet can fall: " + str(self.can_fall(self.pf, state["tetrimino"])))
            pass
            #print("\n")
            #input()
        if len(child_states) > 0:
            #print(state["timer"])
            results = [(s["parent_move"], self.max_node(s, return_type)) for s in child_states]
            best_result = results[0]
            for result in results[1:]:
                if result[1][0] > best_result[1][0]:
                    best_result = result

            if return_type == "path":
                return best_result[1][0], [best_result[0]] + best_result[1][1]
            elif return_type == "tetrimino":
                return best_result[1][0], best_result[1][1]
            else:
                print("you messed up! invalid return type given to max_node!")
        else:
            
            
            #state["tetrimino"].log()
            #print("eval: " + str(eval) + "\n")
            

            if return_type == "path":
                eval = self.greedy_heuristic_evaluation(state["tetrimino"], pf)
                return eval, ["drop"]
            elif return_type == "tetrimino":
                eval = self.greedy_heuristic_evaluation(state["tetrimino"], pf) if self.valid_location(pf, state["tetrimino"]) else - (20 ** 4)
                return eval, state["tetrimino"]
            else:
                print("you messed up! invalid return type given to max_node!")




       


# #==================================================================================================================# #
# #==================================================================================================================# #
# #========================================== EXECUTE BELOW AUTOMATICALLY ===========================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #
