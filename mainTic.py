import numpy as np
import os
import math
import random
import time
import pygame
import matplotlib.pyplot as plt
from tqdm import tqdm

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
    
    def __repr__(self):
        return "TicTacToe"
    
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
          return False
    
        row = action // self.column_count
        column = action % self.column_count
        player = state[row,column]
        
        return (
            np.sum(state[row,:]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
          return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
          return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.reshape(state, (-1))
        return encoded_state

class Node:
    def __init__(self, game, constant, state, parent=None, action_taken=None):
        self.game = game
        self.constant = constant
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        
        self.visit_count = 0
        self.value_sum = 0

    #check if tree is expanded
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    #during selection find the best child with most best ucb
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child

    #calculation for ucb (upper confidence bound)
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.constant * math.sqrt(math.log(self.visit_count) / child.visit_count)

    #expand the child node, by taking a random choice and creating a node from that
    #like each child have it own state to build upon
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)
        
        child = Node(self.game, self.constant, child_state, self, action)
        self.children.append(child)
        return child

    #simulate a game and return value
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        # with Pool(os.cpu_count()-2) as p:
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    
            
            rollout_player = self.game.get_opponent(rollout_player)

                # valid_moves = p.map(self.game.get_valid_moves, rollout_state)
                # valid = [j for x in valid_moves for j in x]
                # action = np.random.choice(where(valid))
                # if count == 0:
                #     args1 = [(rollout_state, action, rollout_player)]
                # else:
                #     args1 = [(rollout_state[0], action, rollout_player)]
                # rollout_state = p.starmap(self.game.get_next_state, args1)
                # args2 = [(rollout_state[0], action)]
                # output = p.starmap(self.game.get_value_and_terminated, args2)
                # value , is_terminal = output[0][0], output[0][1]
                # if is_terminal:
                #     if rollout_player == -1:
                #         value = self.game.get_opponent_value(value)
                #     return value  
                # rollout_player = self.game.get_opponent(rollout_player)
                # count += 1

    #just summing everything up 
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, num_search, constant):
        self.game = game
        self.constant = constant
        self.num_search = num_search
        
    def search(self, state):
        root = Node(self.game, self.constant, state)
        
        for search in (range(self.num_search)):
            node = root

            #check if tree is fully expanded
            while node.is_fully_expanded():
                #pick the best node 
                node = node.select()

            #check if terminal
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            #if not expand leaf and simulate the game
            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            #backpropagate all the value and sum it all up
            node.backpropagate(value)    
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        #get the highest action_probs from 9 choices
        action_probs /= np.sum(action_probs)
        return action_probs

def draw_state(state, X ,Y):
    count = 1
    pygame.draw.line(display_surface, 125, [X//3, 0], [X//3, Y], width=10)
    pygame.draw.line(display_surface, 125, [(X//3)*2, 0], [(X//3)*2, Y], width=10)
    pygame.draw.line(display_surface, 125, [0, Y//3], [X, Y//3], width=10)
    pygame.draw.line(display_surface, 125, [0, (Y//3)*2], [X, (Y//3)*2], width=10)
    for i, row in enumerate(state):
        for j, col in enumerate(row):
            if count == 1:
                zeros = 0
                circZeros = ((X//3)//2) 
            elif count == 2:
                zeros = X//3
                circZeros = ((X//3)//2) * (3)
            else:
                zeros = (X//3) * 2
                circZeros = ((X//3)//2) * (5)
            if j == 0:
                posX1 = [0,zeros]
                posY1 = [(X//3), (Y//3) * count]
                posX2 = [X//3, zeros]
                posY2 = [0, (Y//3) * count]
                center = [((X//3)//2) ,circZeros]
            elif j == 1:
                posX1 = [X//3, zeros ]
                posY1 = [(X//3)*2, (Y//3) * count]
                posX2 = [(X//3)*2, zeros ]
                posY2 = [(X//3), (Y//3) * count]
                center = [((X//3)//2) * (3), circZeros]
            else:
                posX1 = [(X//3)*2, zeros ]
                posY1 = [X, (Y//3) * count]
                posX2 = [X, zeros ]
                posY2 = [(X//3)*2, (Y//3) * count]
                center = [((X//3)//2) * (5), circZeros]
            if col == 1:
                pygame.draw.line(display_surface, (255,0,0), posX1, posY1, width=10)
                pygame.draw.line(display_surface, (255,0,0), posX2,posY2, width=10)
            elif col == -1:
                pygame.draw.circle(display_surface, (0,177,64), center, ((X//3)//2))
        count+= 1
                

if __name__ == "__main__":
    select_mode = int(input("Select [0] for AI VS AI or [1] for Player Vs AI:  "))
    tictactoe = TicTacToe()
    player = 1

    #Human vs AI
    if select_mode == 1:
        pygame.init()

        X = 600
        Y = 600
        white = (255,255,255)
        display_surface = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('TicTacToe')
         
        search_num = int(input("Input amount of search for AI -- (hint: 5000 is decent): "))
        mcts = MCTS(tictactoe, search_num, 1.41)
        state = tictactoe.get_initial_state()
        while True:
            display_surface.fill(white)
            pygame.draw.line(display_surface, 125, [X//3, 0], [X//3, Y], width=10)
            pygame.draw.line(display_surface, 125, [(X//3)*2, 0], [(X//3)*2, Y], width=10)
            pygame.draw.line(display_surface, 125, [0, Y//3], [X, Y//3], width=10)
            pygame.draw.line(display_surface, 125, [0, (Y//3)*2], [X, (Y//3)*2], width=10)
            draw_state(state, X ,Y)
            pygame.event.get()
 
            pygame.display.update()
            if player == 1:
                valid_moves = tictactoe.get_valid_moves(state)
                print("valid_moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
                action = int(input(f"{player}:"))
        
                if valid_moves[action] == 0:
                    print("action not valid")
                    continue
                    
            else:
                neutral_state = tictactoe.change_perspective(state, player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)

            state = tictactoe.get_next_state(state, action, player)
            value, terminal = tictactoe.get_value_and_terminated(state, action)
            
            if terminal:
                if value == 1:
                    if player == 1:
                        print("Congrat! You Won!!")
                    else:
                        print("AI WON")
                else:
                    print("draw")
                break
                
            player = tictactoe.get_opponent(player)

    elif select_mode == 0:
        select_num_game = int(input("pick a number game play: "))
        search_num1 = int(input("Input amount of search for AI[1] -- (hint: 5000 is decent) : "))
        search_num2 = int(input("Input amount of search for AI[2] -- (hint: 5000 is decent) : "))
        mcts = MCTS(tictactoe, search_num1, 1.41)
        mcts2 = MCTS(tictactoe, search_num2, 1.41)
        win1 = 0
        win2 = 0
        draw = 0
        print("playing")
        for i in tqdm(range(select_num_game)):
            state = tictactoe.get_initial_state()
            while True:
                if player == 1:
                    neutral_state = tictactoe.change_perspective(state, player)
                    mcts_probs = mcts.search(neutral_state)
                    #pick the max value from the probability
                    action = np.argmax(mcts_probs)
                else:
                    neutral_state = tictactoe.change_perspective(state, player)
                    mcts_probs = mcts2.search(neutral_state)
                    action = np.argmax(mcts_probs)

                state = tictactoe.get_next_state(state, action, player)
                value, terminal = tictactoe.get_value_and_terminated(state, action)
                
                if terminal:
                    if value == 1:
                        if player == 1:
                            win1 += 1
                        else:
                            win2 += 1
                    else:
                        draw += 1
                    break
                    
                player = tictactoe.get_opponent(player)
        print(f"X wins: {win1} and O wins: {win2} and draw: {draw}" )
        fig, ax = plt.subplots()
        labels = ['X', 'O', 'Draw']
        counts = [win1, win2, draw]
        
        ax.bar(labels, counts)
        
        ax.set_ylabel('Game-played')
        
        plt.show()
    else:
        print("Invalid input please run again")
        exit(1)
        
        