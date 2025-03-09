import numpy as np
import time
from agent import Agent

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "HexPathfinder"
        self.game_start_time = time.time()
        self.total_moves = 0
        self.move_history = []

    def step(self):
        """Make a move on the hex grid"""
        # Check remaining time
        if time.time() - self.game_start_time >= 85:
            return self.free_moves()[0]

        # First move optimization
        if self.total_moves == 0:
            center = self.get_grid_size() // 2
            if self.player_number == 1:  # Vertical connection
                move = [center, center]
            else:  # Horizontal connection
                move = [center, center-1]
            self.set_hex(self.player_number, move)
            self.total_moves += 1
            return move

        best_move = self._find_best_move()
        self.set_hex(self.player_number, best_move)
        self.total_moves += 1
        return best_move

    def _find_best_move(self):
        """Find best move using path and connectivity analysis"""
        available_moves = self.free_moves()
        best_score = float('-inf')
        best_move = available_moves[0]

        for move in available_moves:
            # Try move
            self.set_hex(self.player_number, move)
            
            # Calculate move score
            score = self._evaluate_position(move)
            
            # Check if this move wins
            if self.check_win(self.player_number):
                self.set_hex(0, move)
                return move
                
            # Check if opponent would win next
            self.set_hex(self.adv_number, move)
            if self.check_win(self.adv_number):
                score += 1000  # High priority to blocking moves
                
            # Reset and evaluate
            self.set_hex(self.player_number, move)
            
            # Consider connectivity
            connected_neighbors = sum(1 for neighbor in self.neighbors(move)
                                   if self.get_hex(neighbor) == self.player_number)
            score += connected_neighbors * 2
            
            # Consider path potential
            if self.player_number == 1:  # Vertical
                score += (self.get_grid_size() - abs(move[1] - self.get_grid_size()//2)) * 1.5
            else:  # Horizontal
                score += (self.get_grid_size() - abs(move[0] - self.get_grid_size()//2)) * 1.5
                
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
                
            # Reset for next evaluation
            self.set_hex(0, move)

        return best_move

    def _evaluate_position(self, move):
        """Evaluate position considering paths and connectivity"""
        score = 0
        size = self.get_grid_size()
        
        # Check path development
        if self.player_number == 1:  # Vertical connection
            # Check connection to edges
            has_top = any(self.get_hex([0, y]) == self.player_number 
                         for y in range(size))
            has_bottom = any(self.get_hex([size-1, y]) == self.player_number 
                           for y in range(size))
            score += (has_top + has_bottom) * 10
            
            # Bonus for progress towards goal
            score += (size - abs(move[0] - size//2)) * 2
            
        else:  # Horizontal connection 
            # Check connection to edges
            has_left = any(self.get_hex([x, 0]) == self.player_number 
                          for x in range(size))
            has_right = any(self.get_hex([x, size-1]) == self.player_number 
                           for x in range(size))
            score += (has_left + has_right) * 10
            
            # Bonus for progress towards goal
            score += (size - abs(move[1] - size//2)) * 2
            
        # Check connectivity with existing pieces
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.player_number:
                score += 5
            elif self.get_hex(neighbor) == self.adv_number:
                score += 2  # bonus for blocking opponent
                
        return score

    def update(self, move_other_player):
        """Update game state with opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
        self.move_history.append(move_other_player)