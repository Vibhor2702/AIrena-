import numpy as np
import time
from agent import Agent

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "AdaptiveHexMaster"
        self.game_start_time = time.time()
        self.total_moves = 0
        self.move_history = []
        self.edge_connections = {1: [False, False], 2: [False, False]} # [top/left, bottom/right]
        self.virtual_connections = {}
        
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
            x, y = move
            
            # Check if this move connects the components
            neighbors = self.neighbors(move)
            comp1_connection = any((nx, ny) in comp1 for nx, ny in [tuple(n) for n in neighbors])
            comp2_connection = any((nx, ny) in comp2 for nx, ny in [tuple(n) for n in neighbors])
            
            if comp1_connection and comp2_connection:
                # This move bridges the components - evaluate its quality
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
        
        # Determine game phase (early, mid, late)
        filled_ratio = 1.0 - (len(available_moves) / (self.size * self.size))
        phase = "early" if filled_ratio < 0.3 else "mid" if filled_ratio < 0.7 else "late"

        for move in available_moves:
            # Try move
            self.set_hex(self.player_number, move)
            
            # Base score from position evaluation
            score = self._evaluate_position(move)
                
            # Add connectivity bonus with optimizations for board size
            connected_neighbors = sum(1 for neighbor in self.neighbors(move)
                                   if self.get_hex(neighbor) == self.player_number)
            
            # Adjust connectivity weight by board size and phase
            conn_weight = 4.0 if self.size <= 7 else 3.5 if self.size <= 9 else 3.0
            if phase == "late":
                conn_weight += 1.0  # More important in late game
            score += connected_neighbors * conn_weight
            
            # Virtual connections bonus (second-order connections)
            virtual_conns = self._count_virtual_connections(move)
            score += virtual_conns * 2
            
            # Adaptive path evaluation based on player and board size
            if self.player_number == 1:  # Vertical
                # Column preference - adapt based on board size
                if self.size <= 7:
                    # Small boards: prefer central columns
                    col_score = (self.size - abs(move[1] - self.size//2)) * 1.5
                    score += col_score
                else:
                    # Larger boards: prefer consistent columns with existing pieces
                    col_density = sum(1 for x in range(self.size) if self.get_hex([x, move[1]]) == self.player_number)
                    score += col_density * 2
                
                # Progress toward goal
                if self.edge_connections[1][0] and not self.edge_connections[1][1]:
                    # Connected to top but not bottom - value progress downward
                    score += move[0] * 2  # Reward progress toward bottom
                elif self.edge_connections[1][1] and not self.edge_connections[1][0]:
                    # Connected to bottom but not top - value progress upward
                    score += (self.size - move[0]) * 2  # Reward progress toward top
                elif not self.edge_connections[1][0] and not self.edge_connections[1][1]:
                    # Not connected to either edge yet - first focus on top edge
                    if move[0] == 0:  # Top edge
                        score += 8
                    elif move[0] == self.size-1:  # Bottom edge
                        score += 6
            else:  # Horizontal
                # Row preference - adapt based on board size
                if self.size <= 8:
                    # Small/medium boards: prefer central rows
                    row_score = (self.size - abs(move[0] - self.size//2)) * 1.5
                    score += row_score
                else:
                    # Larger boards: prefer consistent rows with existing pieces
                    row_density = sum(1 for y in range(self.size) if self.get_hex([move[0], y]) == self.player_number)
                    score += row_density * 2.5
                
                # Progress toward goal
                if self.edge_connections[2][0] and not self.edge_connections[2][1]:
                    # Connected to left but not right - value progress rightward
                    score += move[1] * (3 if self.size >= 9 else 2)  # Higher weight on larger boards
                elif self.edge_connections[2][1] and not self.edge_connections[2][0]:
                    # Connected to right but not left - value progress leftward
                    score += (self.size - move[1]) * 2
                elif not self.edge_connections[2][0] and not self.edge_connections[2][1]:
                    # Not connected to either edge yet - first focus on left edge
                    if move[1] == 0:  # Left edge
                        score += 8
                    elif move[1] == self.size-1:  # Right edge
                        score += 6
                
                # Extra horizontal path analysis for larger boards
                if self.size >= 9:
                    # Check for direct path potential
                    path_potential = self._check_path_potential(move, True)
                    score += path_potential * 5
                
            # Defensive evaluation - block opponent's strong moves
            blocking_value = self._evaluate_blocking_value(move)
            
            # Scale blocking value based on phase
            if phase == "early":
                score += blocking_value * 0.7  # Less important early
            elif phase == "mid":
                score += blocking_value * 1.2  # More important mid-game
            else:
                score += blocking_value * 0.9  # Moderate late-game
                
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
                
            # Reset for next evaluation
            self.set_hex(0, move)

        return best_move
        
    def _count_virtual_connections(self, move):
        """Count virtual connections (shared neighbors with friendly pieces)"""
        virtual_count = 0
        my_neighbors = set(tuple(n) for n in self.neighbors(move))
        
        for y in range(self.size):
            for x in range(self.size):
                if self.get_hex([x, y]) == self.player_number and (x,y) != tuple(move):
                    their_neighbors = set(tuple(n) for n in self.neighbors([x, y]))
                    shared = my_neighbors.intersection(their_neighbors)
                    
                    for sx, sy in shared:
                        if self.get_hex([sx, sy]) == 0:  # Empty position creates virtual connection
                            virtual_count += 1
                            
        return virtual_count
        
    def _evaluate_blocking_value(self, move):
        """Evaluate how well a move blocks opponent's strategy"""
        # Try the move as if opponent played it
        self.set_hex(0, move)
        self.set_hex(self.adv_number, move)
        
        # Get baseline opponent connectivity
        opp_connectivity = sum(1 for n in self.neighbors(move) if self.get_hex(n) == self.adv_number)
        
        # Check if it blocks a potential path
        blocking_score = 0
        
        if self.player_number == 1:  # We're vertical, opponent is horizontal
            # Check if this position is on opponent's potential path
            left_connected = False
            right_connected = False
            
            # Check leftward
            x, y = move
            nx = x
            ny = y
            while ny > 0:
                ny -= 1
                if self.get_hex([nx, ny]) == self.adv_number:
                    left_connected = True
                    break
                if self.get_hex([nx, ny]) == self.player_number:
                    break
                    
            # Check rightward
            nx = x
            ny = y
            while ny < self.size - 1:
                ny += 1
                if self.get_hex([nx, ny]) == self.adv_number:
                    right_connected = True
                    break
                if self.get_hex([nx, ny]) == self.player_number:
                    break
                    
            if left_connected and right_connected:
                blocking_score += 10  # High value for blocking a path
                
        else:  # We're horizontal, opponent is vertical
            # Check if this position is on opponent's potential path
            top_connected = False
            bottom_connected = False
            
            # Check upward
            x, y = move
            nx = x
            ny = y
            while nx > 0:
                nx -= 1
                if self.get_hex([nx, ny]) == self.adv_number:
                    top_connected = True
                    break
                if self.get_hex([nx, ny]) == self.player_number:
                    break
                    
            # Check downward
            nx = x
            ny = y
            while nx < self.size - 1:
                nx += 1
                if self.get_hex([nx, ny]) == self.adv_number:
                    bottom_connected = True
                    break
                if self.get_hex([nx, ny]) == self.player_number:
                    break
                    
            if top_connected and bottom_connected:
                blocking_score += 10  # High value for blocking a path
        
        # Reset board
        self.set_hex(0, move)
        
        return blocking_score + opp_connectivity * 2

    def _evaluate_position(self, move):
        """Evaluate position with optimizations for all board sizes"""
        score = 0
        size = self.size
        
        # Edge connection analysis
        if self.player_number == 1:  # Vertical connection
            # Check connection to edges
            edge_value = 15 if size >= 9 else 12
            edge_score = (int(self.edge_connections[1][0]) + int(self.edge_connections[1][1])) * edge_value
            score += edge_score
            
            # Path quality evaluation
            if self.edge_connections[1][0] and not self.edge_connections[1][1]:
                # Top connected - evaluate path toward bottom
                path_quality = self._evaluate_path_quality(move, True, False)  # Downward
                score += path_quality * 4
            elif self.edge_connections[1][1] and not self.edge_connections[1][0]:
                # Bottom connected - evaluate path toward top
                path_quality = self._evaluate_path_quality(move, True, True)  # Upward
                score += path_quality * 4
            
        else:  # Horizontal connection
            # Check connection to edges with size adaptations
            edge_value = 18 if size >= 9 else 15 if size >= 7 else 12
            edge_score = (int(self.edge_connections[2][0]) + int(self.edge_connections[2][1])) * edge_value
            score += edge_score
            
            # Path quality evaluation
            if self.edge_connections[2][0] and not self.edge_connections[2][1]:
                # Left connected - evaluate path toward right
                path_quality = self._evaluate_path_quality(move, False, False)  # Rightward
                score += path_quality * (5 if size >= 9 else 4)
            elif self.edge_connections[2][1] and not self.edge_connections[2][0]:
                # Right connected - evaluate path toward left
                path_quality = self._evaluate_path_quality(move, False, True)  # Leftward
                score += path_quality * 4
            
        # Size-specific position bonuses
        if size <= 7:  # Small boards
            # Value central positions more
            center_dist = abs(move[0] - size//2) + abs(move[1] - size//2)
            center_bonus = (2*size - center_dist) / 2
            score += center_bonus
        elif size <= 9:  # Medium boards
            # Balance between center and edges
            if self.player_number == 1:  # Vertical
                col_centrality = (size - abs(move[1] - size//2)) * 0.8
                score += col_centrality
            else:  # Horizontal
                row_centrality = (size - abs(move[0] - size//2)) * 0.8
                score += row_centrality
        else:  # Large boards
            # Focus more on direct paths than center
            if self.player_number == 1:  # Vertical
                # For vertical - pick good stable columns
                good_cols = [size//3, size//2, 2*size//3]
                if move[1] in good_cols:
                    score += 3
            else:  # Horizontal
                # For horizontal - pick good stable rows
                good_rows = [size//3, size//2, 2*size//3]
                if move[0] in good_rows:
                    score += 3
        
        # Connectivity evaluation
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.player_number:
                score += 5
                # Check for ladder/bridge formations (two connected neighbors that aren't connected to each other)
                for n1 in self.neighbors(move):
                    for n2 in self.neighbors(move):
                        if n1 != n2 and self.get_hex(n1) == self.player_number and self.get_hex(n2) == self.player_number:
                            if n2 not in self.neighbors(n1):  # Not directly connected
                                score += 3  # Bonus for forming a bridge
            elif self.get_hex(neighbor) == self.adv_number:
                score += 1  # Small bonus for blocking opponent
                
        return score
        
    def _evaluate_path_quality(self, move, is_vertical, reverse_direction):
        """Evaluate path quality in specified direction"""
        x, y = move
        
        # Set direction based on parameters
        if is_vertical:
            dx = -1 if reverse_direction else 1
            dy = 0
        else:
            dx = 0
            dy = -1 if reverse_direction else 1
            
        path_score = 0
        nx, ny = x, y
        max_steps = self.size
        steps = 0
        
        while steps < max_steps:
            nx += dx
            ny += dy
            steps += 1
            
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                break  # Hit edge
                
            cell = self.get_hex([nx, ny])
            if cell == self.player_number:
                path_score += 3  # Own piece
            elif cell == 0:
                path_score += 1  # Empty space
            else:
                path_score -= 5  # Opponent piece (bad for path)
                break
                
        return max(0, path_score)  # Don't return negative scores
    
    def _check_path_potential(self, move, is_horizontal):
        """Check path potential with improved connectivity analysis"""
        if is_horizontal:
            # Analyze rightward path potential
            x, y = move
            path_value = 0
            empty_count = 0
            own_count = 0
            
            for ny in range(y+1, self.size):
                cell = self.get_hex([x, ny])
                if cell == self.player_number:
                    own_count += 1
                    path_value += 2
                elif cell == 0:
                    empty_count += 1
                    path_value += 0.5
                else:
                    break  # Blocked
            
            # Bonus for mix of own pieces and open spaces (good path potential)
            if own_count > 0 and empty_count > 0:
                path_value += own_count * empty_count * 0.5
                
            return path_value
        else:
            # Analyze downward path potential
            x, y = move
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
                    break  # Blocked
            
            # Bonus for mix of own pieces and open spaces (good path potential)
            if own_count > 0 and empty_count > 0:
                path_value += own_count * empty_count * 0.5
                
            return path_value

    def update(self, move_other_player):
        """Update game state with opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
        self.move_history.append(move_other_player)