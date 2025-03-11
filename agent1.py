import numpy as np
import time
from agent import Agent

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "OptimizedHexMaster"
        self.game_start_time = time.time()
        self.total_moves = 0
        self.move_history = []
        self.edge_connections = {1: [False, False], 2: [False, False]} # [top/left, bottom/right]
        
    def step(self):
        """Make a move on the hex grid"""
        # Emergency time check
        if time.time() - self.game_start_time >= 85:
            return self.free_moves()[0]
            
        # Update edge connections state
        self._update_edge_connections()
            
        # First move optimization based on board size
        if self.total_moves == 0:
            move = self._get_optimal_first_move()
            self.set_hex(self.player_number, move)
            self.total_moves += 1
            return move
        
        # Win in one move if possible
        for move in self.free_moves():
            self.set_hex(self.player_number, move)
            if self.check_win(self.player_number):
                self.total_moves += 1
                return move
            self.set_hex(0, move)
        
        # Block opponent from winning
        for move in self.free_moves():
            self.set_hex(self.adv_number, move)
            if self.check_win(self.adv_number):
                self.set_hex(self.player_number, move)
                self.total_moves += 1
                return move
            self.set_hex(0, move)
            
        # Look for critical bridge moves
        bridge_move = self._find_bridge_move()
        if bridge_move:
            self.set_hex(self.player_number, bridge_move)
            self.total_moves += 1
            return bridge_move
            
        # Find best move with adaptations for board size and game phase
        best_move = self._find_best_move()
        self.set_hex(self.player_number, best_move)
        self.total_moves += 1
        return best_move
    
    def _get_optimal_first_move(self):
        """Get optimal first move based on board size"""
        size = self.size
        center = size // 2
        
        if self.player_number == 1:  # Vertical (top->bottom)
            if size <= 7:
                return [0, center]  # Start from top edge, center column
            else:
                return [0, center-1]  # Start from top edge, slightly off-center
        else:  # Horizontal (left->right)
            if size <= 7:
                return [center, 0]  # Start from left edge, center row
            elif size == 9:
                return [center, 0]  # Start from left edge at exact center for 9x9
            else:
                return [center-1, 0]  # Start from left edge, slightly above center
                
    def _update_edge_connections(self):
        """Update which edges we're connected to"""
        size = self.size
        
        if self.player_number == 1:  # Vertical (top->bottom)
            self.edge_connections[1][0] = any(self.get_hex([0, y]) == self.player_number for y in range(size))
            self.edge_connections[1][1] = any(self.get_hex([size-1, y]) == self.player_number for y in range(size))
        else:  # Horizontal (left->right)
            self.edge_connections[2][0] = any(self.get_hex([x, 0]) == self.player_number for x in range(size))
            self.edge_connections[2][1] = any(self.get_hex([x, size-1]) == self.player_number for x in range(size))
            
    def _find_bridge_move(self):
        """Find a critical bridge move between components"""
        components = self._find_connected_components()
        
        # If we already have only one component, no bridge needed
        if len(components) <= 1:
            return None
            
        # If we're connected to both edges, prioritize bridges between these components
        if self.edge_connections[self.player_number][0] and self.edge_connections[self.player_number][1]:
            edge0_component = None
            edge1_component = None
            
            # Find components connected to each edge
            for i, comp in enumerate(components):
                if self.player_number == 1:  # Vertical
                    if any(pos[0] == 0 for pos in comp):  # Connected to top
                        edge0_component = i
                    if any(pos[0] == self.size-1 for pos in comp):  # Connected to bottom
                        edge1_component = i
                else:  # Horizontal
                    if any(pos[1] == 0 for pos in comp):  # Connected to left
                        edge0_component = i
                    if any(pos[1] == self.size-1 for pos in comp):  # Connected to right
                        edge1_component = i
            
            # If we found separate components for each edge, prioritize connecting them
            if edge0_component is not None and edge1_component is not None and edge0_component != edge1_component:
                bridge = self._find_best_bridge(components[edge0_component], components[edge1_component])
                if bridge:
                    return bridge
        
        # Otherwise, try to bridge the largest components
        components.sort(key=len, reverse=True)
        if len(components) >= 2:
            bridge = self._find_best_bridge(components[0], components[1])
            if bridge:
                return bridge
                
        return None
        
    def _find_connected_components(self):
        """Find connected components of player's stones"""
        visited = set()
        components = []
        
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.player_number and (x,y) not in visited:
                    component = set()
                    queue = [(x,y)]
                    
                    while queue:
                        px, py = queue.pop(0)
                        if (px,py) in visited:
                            continue
                            
                        visited.add((px,py))
                        component.add((px,py))
                        
                        for nx, ny in [(px+dx, py+dy) for dx, dy in [(0,1),(1,0),(1,-1),(0,-1),(-1,0),(-1,1)]]:
                            if 0 <= nx < self.size and 0 <= ny < self.size and \
                               self.get_hex([nx, ny]) == self.player_number and (nx,ny) not in visited:
                                queue.append((nx,ny))
                    
                    components.append(component)
        
        return components
        
    def _find_best_bridge(self, comp1, comp2):
        """Find the best bridging move between two components"""
        best_bridge = None
        best_score = -1
        
        # Try each free move
        for move in self.free_moves():
            neighbors = self.neighbors(move)
            comp1_conn = any((nx, ny) in comp1 for nx, ny in [tuple(n) for n in neighbors])
            comp2_conn = any((nx, ny) in comp2 for nx, ny in [tuple(n) for n in neighbors])
            
            if comp1_conn and comp2_conn:
                self.set_hex(self.player_number, move)
                score = self._evaluate_position(move)
                self.set_hex(0, move)
                
                if score > best_score:
                    best_score = score
                    best_bridge = move
        
        return best_bridge

    def _find_best_move(self):
        """Find best move with adaptive evaluation based on board size and phase"""
        available_moves = self.free_moves()
        best_score = float('-inf')
        best_move = available_moves[0]
        
        # Special handling for 9x9 as Player 2
        is_size9_p2 = (self.size == 9 and self.player_number == 2)
        
        # Determine game phase (early, mid, late)
        filled_ratio = 1.0 - (len(available_moves) / (self.size * self.size))
        phase = "early" if filled_ratio < 0.3 else "mid" if filled_ratio < 0.7 else "late"

        for move in available_moves:
            self.set_hex(self.player_number, move)
            score = self._evaluate_position(move)
            
            # Add connectivity bonus
            connected_neighbors = sum(1 for n in self.neighbors(move) if self.get_hex(n) == self.player_number)
            conn_weight = 4.0 if self.size <= 7 else 3.5
            score += connected_neighbors * conn_weight
            
            # Adaptive path evaluation
            if self.player_number == 1:  # Vertical
                # Column preference - adapt based on board size
                if self.size <= 7:
                    col_score = (self.size - abs(move[1] - self.size//2)) * 1.5
                    score += col_score
                else:
                    col_density = sum(1 for x in range(self.size) if self.get_hex([x, move[1]]) == self.player_number)
                    score += col_density * 2
                
                # Progress toward goal
                if self.edge_connections[1][0] and not self.edge_connections[1][1]:
                    score += move[0] * 2  # Reward progress toward bottom
                elif self.edge_connections[1][1] and not self.edge_connections[1][0]:
                    score += (self.size - move[0]) * 2  # Reward progress toward top
                elif not self.edge_connections[1][0] and not self.edge_connections[1][1]:
                    if move[0] == 0: score += 8  # Top edge
                    elif move[0] == self.size-1: score += 6  # Bottom edge
            else:  # Horizontal
                # Row preference - adapt based on board size
                if self.size <= 8:
                    row_score = (self.size - abs(move[0] - self.size//2)) * 1.5
                    score += row_score
                else:
                    row_density = sum(1 for y in range(self.size) if self.get_hex([move[0], y]) == self.player_number)
                    score += row_density * 2.5
                
                # Progress toward goal - ENHANCED FOR 9x9
                if self.edge_connections[2][0] and not self.edge_connections[2][1]:
                    # Connected to left but not right - value progress rightward
                    rightward_weight = 6.0 if is_size9_p2 else (3 if self.size >= 9 else 2)
                    score += move[1] * rightward_weight  # Higher weight on 9x9
                elif self.edge_connections[2][1] and not self.edge_connections[2][0]:
                    score += (self.size - move[1]) * 2  # Reward progress toward left
                elif not self.edge_connections[2][0] and not self.edge_connections[2][1]:
                    if move[1] == 0: score += 8  # Left edge
                    elif move[1] == self.size-1: score += 6  # Right edge
                
                # Extra horizontal path analysis - ENHANCED FOR 9x9
                if is_size9_p2:
                    path_potential = self._check_path_potential(move, True)
                    score += path_potential * 8  # Increased from 5
                elif self.size >= 9:
                    path_potential = self._check_path_potential(move, True)
                    score += path_potential * 5
            
            # Defensive evaluation - ENHANCED FOR 9x9
            blocking_weight = 1.6 if is_size9_p2 else (1.2 if phase == "mid" else 0.8)
            score += self._evaluate_blocking_value(move) * blocking_weight
                
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
                
            self.set_hex(0, move)

        return best_move
    
    def _evaluate_blocking_value(self, move):
        """Evaluate how well a move blocks opponent's strategy"""
        # Try the move as if opponent played it
        self.set_hex(0, move)
        self.set_hex(self.adv_number, move)
        
        # Get baseline opponent connectivity
        opp_conn = sum(1 for n in self.neighbors(move) if self.get_hex(n) == self.adv_number)
        blocking_score = opp_conn * 2  # Base blocking value
        
        # Enhanced blocking for 9x9 boards when player 2
        if self.size == 9 and self.player_number == 2:
            # Check if move breaks a critical vertical path
            x, y = move
            top_range = range(0, x)
            bottom_range = range(x+1, self.size)
            
            # Count opponent pieces above and below
            top_pieces = sum(1 for i in top_range if self.get_hex([i, y]) == self.adv_number)
            bottom_pieces = sum(1 for i in bottom_range if self.get_hex([i, y]) == self.adv_number)
            
            # If opponent has pieces on both sides, this is a good blocking move
            if top_pieces > 0 and bottom_pieces > 0:
                blocking_score += 15  # Significantly increase blocking value
        
        # Reset board
        self.set_hex(0, move)
        
        return blocking_score

    def _evaluate_position(self, move):
        """Evaluate position with optimizations for all board sizes"""
        score = 0
        size = self.size
        is_size9_p2 = (size == 9 and self.player_number == 2)
        
        # Edge connection analysis
        if self.player_number == 1:  # Vertical connection
            edge_value = 15 if size >= 9 else 12
            edge_score = (int(self.edge_connections[1][0]) + int(self.edge_connections[1][1])) * edge_value
            score += edge_score
            
            if self.edge_connections[1][0] and not self.edge_connections[1][1]:
                path_quality = self._evaluate_path_quality(move, True, False)  # Downward
                score += path_quality * 4
            elif self.edge_connections[1][1] and not self.edge_connections[1][0]:
                path_quality = self._evaluate_path_quality(move, True, True)  # Upward
                score += path_quality * 4
            
        else:  # Horizontal connection
            # Enhanced for 9x9 boards
            edge_value = 25 if is_size9_p2 else (18 if size >= 9 else 15)
            edge_score = (int(self.edge_connections[2][0]) + int(self.edge_connections[2][1])) * edge_value
            score += edge_score
            
            if self.edge_connections[2][0] and not self.edge_connections[2][1]:
                path_weight = 8 if is_size9_p2 else (5 if size >= 9 else 4)
                path_quality = self._evaluate_path_quality(move, False, False)  # Rightward
                score += path_quality * path_weight
            elif self.edge_connections[2][1] and not self.edge_connections[2][0]:
                path_quality = self._evaluate_path_quality(move, False, True)  # Leftward
                score += path_quality * 4
        
        # Size-specific position bonuses
        if size <= 7:  # Small boards
            center_dist = abs(move[0] - size//2) + abs(move[1] - size//2)
            score += (2*size - center_dist) / 2
        elif size <= 9:  # Medium boards
            if self.player_number == 1:  # Vertical
                score += (size - abs(move[1] - size//2)) * 0.8
            else:  # Horizontal - Enhanced for 9x9
                if is_size9_p2:
                    # For 9x9, prefer exact center row and adjacent rows
                    if move[0] == size//2:
                        score += 5  # Center row bonus
                    elif move[0] == size//2-1 or move[0] == size//2+1:
                        score += 3  # Adjacent row bonus
                else:
                    score += (size - abs(move[0] - size//2)) * 0.8
        
        # Connectivity evaluation
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.player_number:
                neighbor_bonus = 6 if is_size9_p2 else 5  # Enhanced for 9x9
                score += neighbor_bonus
                
                # Check for bridge formations between non-connected neighbors
                if is_size9_p2:  # Enhanced bridge detection for 9x9
                    bridge_count = 0
                    my_neighbors = set(tuple(n) for n in self.neighbors(move))
                    
                    for n1 in my_neighbors:
                        if self.get_hex(list(n1)) == self.player_number:
                            for n2 in my_neighbors:
                                if n1 != n2 and self.get_hex(list(n2)) == self.player_number:
                                    if n2 not in set(tuple(n) for n in self.neighbors(list(n1))):
                                        bridge_count += 1
                    
                    score += bridge_count * 4  # Enhanced bridge bonus
            elif self.get_hex(neighbor) == self.adv_number:
                score += 1
                
        return score
        
    def _evaluate_path_quality(self, move, is_vertical, reverse_direction):
        """Evaluate path quality in specified direction"""
        x, y = move
        dx = (-1 if reverse_direction else 1) if is_vertical else 0
        dy = 0 if is_vertical else (-1 if reverse_direction else 1)
            
        path_score = 0
        nx, ny = x, y
        max_steps = self.size
        steps = 0
        
        while steps < max_steps:
            nx += dx
            ny += dy
            steps += 1
            
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                break
                
            cell = self.get_hex([nx, ny])
            if cell == self.player_number:
                path_score += 3
            elif cell == 0:
                path_score += 1
            else:
                path_score -= 5
                break
                
        return max(0, path_score)
    
    def _check_path_potential(self, move, is_horizontal):
        """Enhanced path potential evaluation"""
        x, y = move
        is_size9_p2 = (self.size == 9 and self.player_number == 2)
        
        if is_horizontal:
            path_value = 0
            empty_count = 0
            own_count = 0
            
            # Enhanced coefficient for 9x9
            empty_value = 1.0 if is_size9_p2 else 0.5
            own_value = 3.0 if is_size9_p2 else 2.0
            
            for ny in range(y+1, self.size):
                cell = self.get_hex([x, ny])
                if cell == self.player_number:
                    own_count += 1
                    path_value += own_value
                elif cell == 0:
                    empty_count += 1
                    path_value += empty_value
                else:
                    break
            
            # Enhanced bonus for 9x9
            if own_count > 0 and empty_count > 0:
                bonus = 1.0 if is_size9_p2 else 0.5
                path_value += own_count * empty_count * bonus
                
            return path_value
        else:
            path_value = 0
            empty_count = 0
            own_count = 0
            
            for nx in range(x+1, self.size):
                cell = self.get_hex([nx, y])
                if cell == self.player_number:
                    own_count += 1
                    path_value += 2
                elif cell == 0:
                    empty_count += 1
                    path_value += 0.5
                else:
                    break
            
            if own_count > 0 and empty_count > 0:
                path_value += own_count * empty_count * 0.5
                
            return path_value

    def update(self, move_other_player):
        """Update game state with opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
        self.move_history.append(move_other_player)