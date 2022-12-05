# #==================================================================================================================# #
# #==================================================================================================================# #
# #==================================================== IMPORTS =====================================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #
import tkinter as tk
from tkinter import Tk, Canvas, Frame, BOTH, StringVar
import numpy as np
from tetrimino import Tetrimino
import tetris_ai
from tetris_ai import TetrisAI
import random
import time
from copy import copy, deepcopy
from time import time, sleep
import math
import csv

# #==================================================================================================================# #
# #==================================================================================================================# #
# #================================================ GLOBAL VARIABLES ================================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #

press_speeds = {
    "Beginner": 3,
    "Intermediate": 4,
    "Advanced": 5,
    "Pro": 6
}

ghost_piece = [[]]

colors = {
    'c': "#0ff",
    'y': "#ff0",
    'p': "#f0f",
    'g': "#0f0",
    'r': "#f00",
    'b': "#00b",
    'o': "#f80",
    ' ': "#fff",
    '~': "#ccc"
}

outline_colors = {
    'c': "#0aa",
    'y': "#aa0",
    'p': "#a0a",
    'g': "#0a0",
    'r': "#a00",
    'b': "#006",
    'o': "#a50",
    ' ': "#aaa",
    '~': "#888" 
}

scoreboard = {}

playfield_width = 10  # width of playfield in # of minos
playfield_height = 20  # height of playfield in # of minos
skybox_height = 4 # height of buffer zone that sits directly above the visible play field in # of minos
playfield_height += skybox_height

bag_size = 1
bag = Tetrimino.make_bag(bag_size)

active_tet = None
next_tet = Tetrimino('L', y=playfield_height - 3)
next_tet.x = (int) (playfield_width/2 - math.floor(next_tet.size / 2.0))

playfield = [[]]

ai = TetrisAI(playfield, active_tet, next_tet, 3, 1)
tet_moves = []

mino_size = 40


mino_border_thickness = 1
window = tk.Tk()
window.title("Tetris Simulation")
window.resizable(False, False)

scoreboard_view = tk.Frame(master=window, padx=25, pady=5)
message_view = tk.Frame(master=window, padx=25, pady=5)

message_text = StringVar(window)
message_label = tk.Label(master=message_view, textvariable = message_text)
message_label.pack(fill=tk.X)

scoreboard_view.grid_columnconfigure(0, weight=1)
scoreboard_view.grid_columnconfigure(1, weight=1)
scoreboard_view.grid_columnconfigure(2, weight=1)

level_label = StringVar(window)
clears_label = StringVar(window)
score_label = StringVar(window) 

level_view = tk.Label(master=scoreboard_view, textvariable=level_label)
level_view.grid(row=0, column=0)

clears_view = tk.Label(master=scoreboard_view, textvariable=clears_label)
clears_view.grid(row=0, column=1)

score_view = tk.Label(master=scoreboard_view, textvariable=score_label)
score_view.grid(row=0, column=2)

scoreboard_view.pack(fill=tk.X)
message_view.pack(fill=tk.X)

playfield_view = tk.Canvas(
    window, width=playfield_width * mino_size, height=playfield_height * mino_size)
playfield_view.pack(side="left")

queue_view = tk.Canvas(master=window, width=4 * mino_size,
                       height=4 * mino_size, bo=2)
queue_view.pack(side="right", anchor="n")


paused = False
prev_time = -1

# #==================================================================================================================# #
# #==================================================================================================================# #
# #================================================ HELPER FUNCTIONS ================================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #

def clears_needed():
    return scoreboard["level"] * ((5 + scoreboard["level"] * 5) / 2) 

def generate_pf():
    global playfield
    playfield = [[' ' for x in range(playfield_width)]
             for y in range(playfield_height)]
    return playfield
playfield = generate_pf()

def spawn_new_tetrimino():
    global active_tet, next_tet, ai, bag, playfield
    active_tet = next_tet
    next_tet = Tetrimino(bag.pop(0), x= 4, y=playfield_height - 3)
    
    # if game is over because spawn tetrimino is blocked out 
    if not TetrisAI.valid_location(playfield, active_tet):
        scoreboard["game_over"] = True
        return "spawn_new_tetrimino() exited"
    if not TetrisAI.tetrimino_has_landed(playfield, active_tet):
        active_tet.y -= 1
    ai.pf = playfield
    ai.tet = active_tet
    ai.next_tet = next_tet
    if not bag:
        bag = Tetrimino.make_bag(bag_size)
spawn_new_tetrimino()

def generate_scoreboard():
    global scoreboard
    scoreboard = {
        "time_elapsed": 0,
        "score": 0, 
        "level": 1,
        "clears": 0,
        "game_over": False
    }
generate_scoreboard()
# #==================================================================================================================# #
# #==================================================================================================================# #
# #============================================== TETRIS SIM FUNCTIONS ==============================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #

def draw_mino(x, y, c):
    c = c.lower()
    # print("x: " + str(x) + "; y: " + str(y))
    global playfield, playfield_view
    if y > playfield_height - 4 and c == ' ':
        playfield_view.create_rectangle((x-1)*mino_size, (playfield_height-y)*mino_size, x*mino_size - mino_border_thickness, (playfield_height - y+1)*mino_size - mino_border_thickness,
        outline=outline_colors[c], fill="#222", width=mino_border_thickness)
    elif x > 0 and y > 0 and y <= playfield_height and x <= playfield_width:
        playfield_view.create_rectangle((x-1)*mino_size, (playfield_height-y)*mino_size, x*mino_size - mino_border_thickness, (playfield_height - y+1)*mino_size - mino_border_thickness,
        outline=outline_colors[c], fill=colors[c], width=mino_border_thickness)
    else:
        print("ERROR - Invalid x or y position given. Valid values: x = [1, " + str(
            playfield_width) + "] and y = [1, " + str(playfield_height) + "]")

def draw_queue():
    global queue_view, next_tet
    queue_view.delete("all")
    nt = deepcopy(next_tet)
    queue_view.update()
    nt.minos = [row for row in nt.minos if row.count(' ') != len(row)]
    Ox = (queue_view.winfo_width() - mino_size * len(nt.minos[0])) / 2
    Oy = (queue_view.winfo_height() - mino_size * len(nt.minos)) / 2
    for row in range(len(nt.minos)):
        for col in range(len(nt.minos[0])):
            if nt.minos[-row-1][col] != ' ': # and row <= playfield_height-4:
                queue_view.create_rectangle(
                    Ox + col * mino_size,
                    Oy + row * mino_size,
                    Ox + (col + 1) * mino_size - mino_border_thickness,
                    Oy + (row + 1) * mino_size - mino_border_thickness,
                    outline=outline_colors[nt.minos[-row-1][col]],
                    fill=colors[nt.minos[-row-1][col]],
                    width=mino_border_thickness)

def log_pf():
    for i in range(len(playfield)):
        print("".join(playfield[-i-1]))

def draw_playfield():
    global playfield_view 
    playfield_view.delete("all")
    ghost_tet = deepcopy(active_tet)
    #print("active tet: x= " + str(active_tet.x) + ", y= " + str(active_tet.y))
    ai.drop_tetrimino(playfield, ghost_tet)
    #print("ghost after drop: x= " + str(ghost_tet.x) + ", y= " + str(ghost_tet.y))
    ghost_tet.minos = [['~' if mino != ' ' else ' ' for mino in row] for row in ghost_tet.minos]
    for row in range(playfield_height):
        for col in range(playfield_width):
            draw_mino(col + 1, row + 1, playfield[row][col])
    if active_tet.ldml != 0:
        draw_tetrimino(ghost_tet)
        draw_tetrimino(active_tet)

def draw_tetrimino(tet):
    for row in range(tet.size):
        for col in range(tet.size):
            if tet.minos[row][col] != ' ': # and row <= playfield_height-4:
                draw_mino(tet.x + col, tet.y+row, tet.minos[row][col])

def update_scoreboard():
    global level_label, score_label, clears_label
    level_label.set("LEVEL: " + str(scoreboard["level"]))
    clears_label.set("CLEARS LEFT: " + str(clears_needed() - scoreboard["clears"]))
    score_label.set("SCORE: "+ str(scoreboard["score"]))

def rotate_ccw(event):
    global active_tet, playfield
    ai.rotate_tetrimino(playfield, active_tet, "ccw")
    update_window()

def rotate_cw(event):
    global active_tet, playfield
    ai.rotate_tetrimino(playfield, active_tet, "cw")
    update_window()

def move_left(event):
    global active_tet, playfield
    ai.move_tetrimino(playfield, active_tet, "l")
    update_window()

def move_right(event):
    global active_tet, playfield
    ai.move_tetrimino(playfield, active_tet, "r")
    update_window()

def drop(event):
    global active_tet, playfield
   
    if not ai.tetrimino_has_landed(playfield, active_tet):
        ai.drop_tetrimino(playfield, active_tet)
        update()

def fall(event):
    global active_tet, playfield
    t = deepcopy(active_tet)
    t.y -= 2
    
    if not active_tet.landed and ai.valid_location(playfield, t):
        active_tet.y -= 1
        active_tet.reset_LDML()
        update_window()

def advance_sim(event):
    global paused, active_tet, playfield
    if paused:
        print("\nUNPAUSED\n")
        paused = False    

def stop_game():
    playfield_view.delete("all")
    window.quit()

def update_window():
    global window, playfield_view
    draw_queue()
    draw_playfield()
    update_scoreboard()
    window.update_idletasks()
    #window.update()

def update(debug = True, graphics = False):
    global scoreboard, active_tet, ai, next_tet, playfield, bag, message_text
    
    scoreboard["clears"] += (clears := ai.update_playfield(playfield, active_tet))
    scoreboard["score"] += ai.score_move(clears, scoreboard["level"])
    if graphics:
        update_window()
        sleep(0.2)
        message_text.set("")
   
    if scoreboard["clears"] >= clears_needed():
        #print("LEVEL " + str(scoreboard["level"]) + " CLEARED! Score is ", scoreboard["score"])
        if graphics:
            message_text.set("LEVEL " + str(scoreboard["level"]) + " CLEARED!")
        bag = Tetrimino.make_bag(bag_size)        
        scoreboard["level"] += 1

    spawn_new_tetrimino()
    scoreboard["game_over"] = TetrisAI.game_over(playfield)


    
def game_loop():
    global active_tet, ai, next_tet, playfield, bag, scoreboard, window, prev_time

    # apply gravity 
    if prev_time != (p:= ai.gravity(playfield, active_tet, prev_time, time(), scoreboard["level"])):
        prev_time = p
        update_window()
            
    
    # check if active tetrimino has landed
    if active_tet.ld_timer == -1 and ai.tetrimino_has_landed(playfield, active_tet):
        # if so, check if active tetrimino needs locked down
        active_tet.ld_timer = time()
    elif ai.tetrimino_locked_down(playfield, active_tet, time()):
        active_tet.ld_timer  = -1
        update()

    if not scoreboard["game_over"]:
        # continue looping game logic
        window.after(1, game_loop)












def sim_loop():
    global paused, active_tet, ai, next_tet, playfield, bag, message_text, scoreboard, window, prev_time, tet_moves
    
    paused = False
    
    if not paused:
        m = ""
        if tet_moves:
            TetrisAI.execute_move(playfield, active_tet, (m:= tet_moves[0]))
            print("move: " + m)
            tet_moves.pop(0)
            # apply gravity 
            if prev_time != (p:= ai.gravity(playfield, active_tet, prev_time, scoreboard["time_elapsed"], scoreboard["level"])):
                prev_time = p
                update_window()
    
        # check if active tetrimino has landed
        if m == "drop":
            update()
            active_tet.ld_timer  = -1
            ai_results = ai.get_best_moves()
            #print(ai_results[1])
            tet_moves = ai.get_simplified_path(ai_results[1])
            #print(tet_moves)
            print("eval: " + str(ai_results[0]))
            
            paused = True

    if not scoreboard["game_over"]:
        scoreboard["time_elapsed"] += 1.0 / ai.mps
        # continue looping game logic
        window.after(20, sim_loop)
        
    else:
        message_text.set("Game over!")





def log_scoreboard():
    global scoreboard
    s = "\r|| "
    s += "score: " + str(scoreboard["score"]).ljust(10)
    s += "level: " + str(scoreboard["level"]).ljust(4)
    s += "clears: " + (str(scoreboard["clears"]) +"/"+str(round(clears_needed()))).ljust(9) + " ||"
    print(s, end='\r')



def run_trial(AI, debug = False, graphics = False,):
    global paused, active_tet, ai, next_tet, playfield, bag, message_text, scoreboard, window, prev_time, tet_moves
    ai = AI
    generate_scoreboard()
    playfield = generate_pf()
    bag = Tetrimino.make_bag(bag_size)
    next_tet = Tetrimino(bag.pop(), 4, 21)
    spawn_new_tetrimino()
    tet_moves = ai.get_simplified_path(ai.get_best_moves()[1])
    

    while not scoreboard["game_over"]:
        m = ""
        if tet_moves:
            TetrisAI.execute_move(playfield, active_tet, (m:= tet_moves[0]))
            tet_moves.pop(0)
            # apply gravity 
            if prev_time != (p:= ai.gravity(playfield, active_tet, prev_time, scoreboard["time_elapsed"], scoreboard["level"])):
                prev_time = p
                if graphics:
                    update_window()

        # check if active tetrimino has landed
        if m == "drop":
            
            update(debug, graphics)

            if not scoreboard["game_over"]:

                active_tet.ld_timer  = -1
                ai_results = ai.get_best_moves()
                
                tet_moves = ai.get_simplified_path(ai_results[1])
                if debug:
                    print(tet_moves)
                    print("eval: " + str(ai_results[0]))
        if not tet_moves and m != "drop":
            print("huh?")
        scoreboard["time_elapsed"] += 1.0 / ai.mps
        log_scoreboard()        
    if graphics:
        message_text.set("Game over!")
    print()
    return scoreboard    




def write_batch_csv(batch_results, fn="batches.csv"):
    fields = [
        "batch_size",
        "method",
        "score",
        "level",
        "clears",
        "time",
        "mps",
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
    with open(fn, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields)
        writer.writerow(batch_results)  


# batch size is the number of trials to run using the same weights
def run_trials(batch_size, num_batches, method, debug=False, graphics=False):
    global paused, active_tet, ai, next_tet, playfield, bag, message_text, scoreboard, window, prev_time, tet_moves, next_tet
    while True:
        for n in range(num_batches):
            batch_stats = {
                "batch_size": batch_size,
                "method": method,
                "score": 0 ,
                "level" : 0.0 ,
                "clears" : 0 ,
                "time" : 0.0 ,
                "mps" : 0.0  
            }
            params = {param : random.uniform(*TetrisAI.param_weight_bounds[param]) for param in TetrisAI.param_labels}
            for trial in range(batch_size):
                start = time()
                mps = press_speeds["Intermediate"]
                ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method, params = params)
                print("-"*51)
                print("|| " + ("Trial " + str(trial+1) + " - Batch " + str(n+1)).center(45) + " ||" )
                print("-"*51)
                run_trial(ai, debug, graphics)
                
                batch_stats["mps"] += mps  * (1.0 / batch_size)
                batch_stats["time"] += (time() - start) * (1.0 / batch_size)
                for stat in scoreboard:
                    if stat in batch_stats:
                        batch_stats[stat] += scoreboard[stat] * (1.0 / batch_size)
            batch_stats.update(ai.params)
            #print(batch_stats)
            write_batch_csv(batch_stats)
            print("\n" + "-"*50 + "\n\n")
            for key in batch_stats:
                print(key + ": " +  str(batch_stats[key]) )
            print("\n\n" + "-"*50 + "\n")
        #ui = input("Would you like to continue running batches, or quit?")
        #if ui.lower() in ("q", "n", "no", "exit", "quit"):
        #    break
        break

def recover_last_batch(method, fn="batches.csv"):
    fields = [
        "batch_size",
        "method",
        "score",
        "level",
        "clears",
        "time",
        "mps",
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
    with open(fn, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        for result in reversed(list(reader)):
            if result["method"] == method:
                batch = {item: float(result[item]) for item in result if item != "method"}
                batch.update({"method": method})
                return batch
# gets last used set of param weights from csv file
def recover_last_results(method, fn="batches.csv"):
    result = recover_last_batch(method)
    return float(result["score"]), {item: float(result[item]) for item in result if item in TetrisAI.param_labels}


def train(batch_size, epochs, max_delta, learning_rate, method, debug=False, graphics=False, new_weights = False, params = {}, initial_batch_size = -1, score = 0, start_batch_stats = {}):
    batch_stats = {
        "batch_size": batch_size,
        "method": method,
        "score": 0 ,
        "level" : 0.0 ,
        "clears" : 0 ,
        "time" : 0.0 ,
        "mps" : 0.0  
    }
    if start_batch_stats:
        batch_stats = {
            "batch_size": start_batch_stats["batch_size"],
            "method": start_batch_stats["method"],
            "score": start_batch_stats["score"],
            "level" :start_batch_stats["level"] ,
            "clears" : start_batch_stats["clears"] ,
            "time" : start_batch_stats["time"] ,
            "mps" : start_batch_stats["mps"] ,  
        }
        score = start_batch_stats["score"]
    if new_weights:
        write_batch_csv({})
        params = {param : random.uniform(*TetrisAI.param_weight_bounds[param]) for param in TetrisAI.param_labels}
        print(params)
    elif params:
        pass
    else:       
        if initial_batch_size != -1:
            batch_stats = recover_last_batch(method)
            params = recover_last_results(method)[1]
            old_bs = batch_stats["batch_size"]
            ratio = (old_bs/float(old_bs + initial_batch_size))
            batch_stats = {
                "batch_size": old_bs + initial_batch_size,
                "method": method,
                "score": ratio * batch_stats["score"],
                "level" : ratio* batch_stats["level"] ,
                "clears" : ratio * batch_stats["clears"],
                "time" : ratio * batch_stats["time"],
                "mps" : 0.0  
            }
            print("-"*51)
            print("|| " + ("Re-Running W/ Old Weights:").center(45) + " ||" )
            for trial in range(initial_batch_size):
                start = time()
                mps = press_speeds["Intermediate"]
                #print(params)
                ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method, params = params)
                print("-"*51)
                run_trial(ai, debug, graphics)
                
                batch_stats["mps"] += mps  * (1.0 / batch_stats["batch_size"])
                batch_stats["time"] += (time() - start) * (1.0 / batch_stats["batch_size"])
                for stat in scoreboard:
                    if stat in batch_stats:
                        batch_stats[stat] += scoreboard[stat] * (1.0 / batch_stats["batch_size"])
            batch_stats.update(params)
            write_batch_csv(batch_stats)
            print("\n" + "-"*50 + "\n\n")
            for key in batch_stats:
                print(key + ": " +  str(batch_stats[key]) )
            print("\n\n" + "-"*50 + "\n")
            score = batch_stats["score"]
        else:
            score, params = recover_last_results(method);
    if score == 0:
        bs = batch_size
        if initial_batch_size != -1:
            bs = initial_batch_size
        batch_stats = {
            "batch_size": bs,
            "method": method,
            "score": 0 ,
            "level" : 0.0 ,
            "clears" : 0 ,
            "time" : 0.0 ,
            "mps" : 0.0  
        }
        print("-"*51)
        print("|| " + ("Running Initial Trials").center(45) + " ||" )
        for trial in range(bs):
            start = time()
            mps = press_speeds["Intermediate"]
            print(params)
            ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method, params = params)
            print("-"*51)
            run_trial(ai, debug, graphics)
            
            batch_stats["mps"] += mps  * (1.0 / bs)
            batch_stats["time"] += (time() - start) * (1.0 / bs)
            for stat in scoreboard:
                if stat in batch_stats:
                    batch_stats[stat] += scoreboard[stat] * (1.0 / bs)
        batch_stats.update(params)
        write_batch_csv(batch_stats)
        print("\n" + "-"*50 + "\n\n")
        for key in batch_stats:
            print(key + ": " +  str(batch_stats[key]) )
        print("\n\n" + "-"*50 + "\n")
        score = batch_stats["score"]
    

    for n in range(epochs):
        temp_batch_stats = {
            "batch_size": batch_size,
            "method": method,
            "score": 0 ,
            "level" : 0.0 ,
            "clears" : 0 ,
            "time" : 0.0 ,
            "mps" : 0.0  
        }
        # generate weight modifications
        weight_mods = { param: random.uniform(*TetrisAI.param_weight_bounds[param]) * max_delta * random.choice((-1,1)) for param in params }
        temp_params = { param: params[param] + weight_mods[param] for param in params}
        for param in temp_params:
            #print(param, temp_params[param], TetrisAI.param_weight_bounds[param][0], TetrisAI.param_weight_bounds[param][1])
            if temp_params[param] > TetrisAI.param_weight_bounds[param][1]:
                temp_params[param] = TetrisAI.param_weight_bounds[param][1]
            elif temp_params[param] < TetrisAI.param_weight_bounds[param][0]:
                temp_params[param] = TetrisAI.param_weight_bounds[param][0]
        # test out performance of new weights
        print("-"*51)
        print("|| " + ("Testing Temp Weights - Epoch " + str(n+1)).center(45) + " ||" )
        for trial in range(batch_size):
            start = time()
            mps = press_speeds["Intermediate"]
            ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method, params = temp_params)
            
            print("-"*51)
            run_trial(ai, debug, graphics)
            
            temp_batch_stats["mps"] += mps  * (1.0 / batch_size)
            temp_batch_stats["time"] += (time() - start) * (1.0 / batch_size)
            for stat in scoreboard:
                if stat in temp_batch_stats:
                    temp_batch_stats[stat] += scoreboard[stat] * (1.0 / batch_size)
        temp_batch_stats.update(temp_params)
        write_batch_csv(temp_batch_stats)
        print("\n" + "-"*50 + "\n\n")
        for key in temp_batch_stats:
            print(key + ": " +  str(temp_batch_stats[key]) )
        print("\n\n" + "-"*50 + "\n")

        # modify weights based on performance of temp weights
        if temp_batch_stats["score"] > score:
            mult = (temp_batch_stats["score"] - score)/float(score + 1)
        

            if mult > 1:
                mult = 1
            elif mult < -1:
                mult = -1
            
            
            for param in params:
                params[param] += float(mult) * float(weight_mods[param]) * float(learning_rate)


            batch_stats = {
                "batch_size": batch_size,
                "method": method,
                "score": 0 ,
                "level" : 0.0 ,
                "clears" : 0 ,
                "time" : 0.0 ,
                "mps" : 0.0  
            }
            print("-"*51)
            print("|| " + ("New Modified Weights - Epoch " + str(n+1)).center(45) + " ||" )
                
            for trial in range(batch_size):
                start = time()
                mps = press_speeds["Intermediate"]
                ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method, params = temp_params)
                print("-"*51)
                run_trial(ai, debug, graphics)
                
                batch_stats["mps"] += mps  * (1.0 / batch_size)
                batch_stats["time"] += (time() - start) * (1.0 / batch_size)
                for stat in scoreboard:
                    if stat in batch_stats:
                        batch_stats[stat] += scoreboard[stat] * (1.0 / batch_size)
            #print(temp_params)
            batch_stats.update(temp_params)
            #print(batch_stats)
            #print(batch_stats)
            write_batch_csv(batch_stats)
            print("\n" + "-"*50 + "\n\n")
            for key in batch_stats:
                print(key + ": " +  str(batch_stats[key]) )
            print("\n\n" + "-"*50 + "\n")
            score = batch_stats["score"]
        
        # run trials on old weights again to improve the accuracy of it's avg score
        elif batch_stats["batch_size"] < 50:
            old_bs = batch_stats["batch_size"]
            ratio = (old_bs/float(old_bs + batch_size))
            if ratio == 0:
                print(ratio)
            batch_stats = {
                "batch_size": old_bs + batch_size,
                "method": method,
                "score": ratio * batch_stats["score"],
                "level" : ratio* batch_stats["level"] ,
                "clears" : ratio * batch_stats["clears"],
                "time" : ratio * batch_stats["time"],
                "mps" : ratio * batch_stats["mps"],  
            }
            print("-"*51)
            print("|| " + ("Re-Running W/ Old Weights: Epoch " + str(n+1)).center(45) + " ||" )
            for trial in range(batch_size):
                start = time()
                mps = press_speeds["Intermediate"]
                ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method, params = params)
                print("-"*51)
                run_trial(ai, debug, graphics)
                
                batch_stats["mps"] += mps  * (1.0 / batch_stats["batch_size"])
                batch_stats["time"] += (time() - start) * (1.0 / batch_stats["batch_size"])
                for stat in scoreboard:
                    if stat in batch_stats:
                        batch_stats[stat] += scoreboard[stat] * (1.0 / batch_stats["batch_size"])
            batch_stats.update(params)
            write_batch_csv(batch_stats)
            print("\n" + "-"*50 + "\n\n")
            for key in batch_stats:
                print(key + ": " +  str(batch_stats[key]) )
            print("\n\n" + "-"*50 + "\n")
            score = batch_stats["score"] 

# #==================================================================================================================# #
# #==================================================================================================================# #
# #========================================== EXECUTE BELOW AUTOMATICALLY ===========================================# #
# #==================================================================================================================# #
# #==================================================================================================================# #

"""
tet = Tetrimino('I', -1, 1)
tet.log()
for i in range(4):
    ai.rotate_tetrimino(playfield, tet, "l")
    tet.log()

tet = Tetrimino('I', -1, 1)
tet.log()
for i in range(4):
    ai.rotate_tetrimino(playfield, tet, "r")
    tet.log()
    """

"""
tet = ai.rotate_tetrimino(playfield, Tetrimino('I', 1, 1), "r")
tet = ai.drop_tetrimino(playfield, tet)
tet.x = -1


draw_tetrimino(tet)
update_window()
input()

tet = ai.rotate_tetrimino(playfield, tet, "r")

draw_tetrimino(tet)
update_window()
input() """


window.bind("<n>", advance_sim)
window.bind("<Return>", advance_sim)
window.bind("<N>", advance_sim)
"""
window.bind("<Up>", rotate_ccw)
window.bind("<Right>", move_right)
window.bind("<Down>", rotate_cw)
window.bind("<Left>", move_left)
window.bind("<d>", drop)
window.bind("<f>", fall)
window.bind("<D>", drop) 
window.bind("<F>", fall)
window.after(100, game_loop)
window.mainloop()
"""

#window.bind("<space>", drop)
#window.bind("<Shift_L>", fall)
#window.bind("<Shift_R>", fall)
#window.bind("<Next>", drop)
"""bag = Tetrimino.make_bag(3)
for t in bag:
    Tetrimino(t).log() """


"""
active_tet = Tetrimino('T', 3, 2)
ai = TetrisAI(playfield, active_tet, next_tet, 3, 1)
print(r:=ai.get_best_moves())
print(ai.get_simplified_path(r[1]))
print(ai.get_simplified_path(["fall", "fall", "left", "ccw", "ccw", "fall", "drop"]))
"""


"""
print(ai.move_tetrimino(playfield, Tetrimino('t', 1, 0), "l"))
print(ai.move_tetrimino(playfield, Tetrimino('t', 1, 0), "r")) """

"""
active_tet = Tetrimino('S', 2, 2)
TetrisAI.drop_tetrimino(playfield, active_tet)
TetrisAI.update_playfield(playfield, active_tet)
active_tet = Tetrimino('T', 6, 6)



ai = TetrisAI(playfield, active_tet, next_tet, press_speeds["Beginner"], scoreboard["level"])
#TetrisAI.rotate_tetrimino(playfield, active_tet, "cw")
tet_moves = ai.get_simplified_path((r:=ai.get_best_moves())[1])
print(tet_moves)
print(r[0])
paused = True
update_window() """


"""
ai = TetrisAI(playfield, active_tet, next_tet, 3, 1, "yield_to_next")
ai_results = ai.get_best_moves()
print(ai_results[1])
tet_moves = ai.get_simplified_path(ai_results[1])
print(tet_moves)
print("eval: " + str(ai_results[0]))


window.after(100, sim_loop)
window.mainloop()
"""
#run_trials(3, 3, method="yield_to_next", debug=False, graphics=False)
#run_trials(10, 20, method="greedy", debug=False, graphics=False)


user_weights = {
    "cliff_l_e" : 5 ,  # P0 
    "cliff_h_e" : 2 , # P1
    "cliff_l_w" : -10 , # P2
    "cliff_h_w" : -1 , # P3
    "m_stack_w" : -2 , # P4
    "m_stack_e" : 2 , # P5
    "a_stack_w" : -2 , # P6
    "a_stack_e" : 1 , # P7
    "stack_d_w" : 0 , # P8
    "stack_d_e" : 3 , # P9
    "stack_d_thresh" : 15 , # P10
    "score_w" : 10000, # P11
    "go_w" : -10 , # P12
}



params = recover_last_results("greedy")[1]
train(5, 1000, .1, .25, "yield_to_next", graphics=False, params=params, start_batch_stats= recover_last_batch("greedy"))
#train(5, 1000, .1, .25, "greedy", graphics=False)