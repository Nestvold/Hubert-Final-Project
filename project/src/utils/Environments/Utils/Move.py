from Utils import Busted
from random import random, randint


def Move(pos, action, grid, enemies=set(), pMM=0.0, MM=False, Fans=False, rocket_motor=False):
    solids = set([2, 3])

    match action:
        case 0:
            return Move_Left(pos, grid, solids, enemies, pMM, MM, Fans)
        case 1:
            return Move_Jump(pos, grid, solids, enemies, pMM, MM, Fans, rocket_motor)
        case 2:
            return Move_Right(pos, grid, solids, enemies, pMM, MM, Fans)
        case _:
            print('Action:', action)
            return None


def Move_Left(new_pos, grid, solids, enemies, pMM, MM, Fans):
    R = -1
    
    if grid[new_pos[0], new_pos[1]-1] != 2:                      # Check if he can move left
        new_pos = (new_pos[0], new_pos[1]-1)                     # Move left

        if Busted(enemies, new_pos, pMM):
            if MM:
                return True, new_pos, R
            if Fans:
                R = -(randint(20, 100))
        
        if grid[new_pos[0]+1, new_pos[1]] not in solids:         # Check if he can fall
            new_pos = (new_pos[0]+1, new_pos[1])                 # Fall down
            
            if Busted(enemies, new_pos, pMM):
                if MM:
                    return True, new_pos, R
                if Fans:
                    R = -(randint(20, 100))
            
            if grid[new_pos[0], new_pos[1]-1] != 2:              # Check if he can move left
                new_pos = (new_pos[0], new_pos[1]-1)             # Move left 

                if Busted(enemies, new_pos, pMM):
                    if MM:
                        return True, new_pos, R
                    if Fans:
                        R = -(randint(20, 100))
                
                if grid[new_pos[0]+1, new_pos[1]] not in solids:  # Check if he can fall
                    new_pos = (new_pos[0]+1, new_pos[1])          # Fall down
                    if MM:
                        return True, new_pos, R
                    if Fans:
                        R = -(randint(20, 100))
            else:
                if grid[new_pos[0]+1, new_pos[1]] not in solids:  # Check if he can fall
                    new_pos = (new_pos[0]+1, new_pos[1])          # Fall down

                    if Busted(enemies, new_pos, pMM):
                        if MM:
                            return True, new_pos, R
                        if Fans:
                            R = -(randint(20, 100))
    else: 
        if grid[new_pos[0]+1, new_pos[1]] not in solids:         # Check if he can fall
            new_pos = (new_pos[0]+1, new_pos[1])                 # Fall down

            if Busted(enemies, new_pos, pMM):
                if MM:
                    return True, new_pos, R
                if Fans:
                    R = -(randint(20, 100))

            if grid[new_pos[0]+1, new_pos[1]] not in solids:     # Check if he can fall
                new_pos = (new_pos[0]+1, new_pos[1])             # Fall down

                if Busted(enemies, new_pos, pMM):
                    return True, new_pos, R

    return False, new_pos, R


def Move_Right(new_pos, grid, solids, enemies, pMM, MM, Fans):
    R = -1
    
    if grid[new_pos[0], new_pos[1]+1] != 2:                      # Check if he can move right
        new_pos = (new_pos[0], new_pos[1]+1)                     # Move right 
        
        if Busted(enemies, new_pos, pMM):
            if MM:
                return True, new_pos, R
            if Fans:
                R = -(randint(20, 100))
        
        if grid[new_pos[0]+1, new_pos[1]] not in solids:         # Check if he can fall
            new_pos = (new_pos[0]+1, new_pos[1])                 # Fall down
            
            if Busted(enemies, new_pos, pMM):
                if MM:
                    return True, new_pos, R
                if Fans:
                    R = -(randint(20, 100))
            
            if grid[new_pos[0], new_pos[1]+1] != 2:              # Check if he can move right
                new_pos = (new_pos[0], new_pos[1]+1)             # Move right 

                if Busted(enemies, new_pos, pMM):
                    if MM:
                        return True, new_pos, R
                    if Fans:
                        R = -(randint(20, 100))
                
                if grid[new_pos[0]+1, new_pos[1]] not in solids:  # Check if he can fall
                    new_pos = (new_pos[0]+1, new_pos[1])          # Fall down

                    if Busted(enemies, new_pos, pMM):
                        if MM:
                            return True, new_pos, R
                        if Fans:
                            R = -(randint(20, 100))
            else:
                if grid[new_pos[0]+1, new_pos[1]] not in solids:  # Check if he can fall
                    new_pos = (new_pos[0]+1, new_pos[1])          # Fall down

                    if Busted(enemies, new_pos, pMM):
                        if MM:
                            return True, new_pos, R
                        if Fans:
                            R = -(randint(20, 100))
    else: 
        if grid[new_pos[0]+1, new_pos[1]] not in solids:         # Check if he can fall
            new_pos = (new_pos[0]+1, new_pos[1])                 # Fall down

            if Busted(enemies, new_pos, pMM):
                if MM:
                    return True, new_pos, R
                if Fans:
                    R = -(randint(20, 100))

            if grid[new_pos[0]+1, new_pos[1]] not in solids:     # Check if he can fall
                new_pos = (new_pos[0]+1, new_pos[1])             # Fall down

                if Busted(enemies, new_pos, pMM):
                    if MM:
                        return True, new_pos, R
                    if Fans:
                        R = -(randint(20, 100))

    return False, new_pos, R


def Move_Jump(new_pos, grid, solids, enemies, pMM, MM, Fans, rocket_motor):
    R = -5
    
    if grid[new_pos[0] + 1, new_pos[1]] in solids:               # Checks if he stands on a surface
        if grid[new_pos[0] - 1, new_pos[1]] != 2:                # Checks if he can jump
            new_pos = (new_pos[0] - 1, new_pos[1])               # He jumps
            
            if Busted(enemies, new_pos, pMM):
                if MM:
                    return True, new_pos, R
                if Fans:
                    R = -(randint(20, 100))         

            if grid[new_pos[0] - 1, new_pos[1]] != 2:            # Checks if he will hit a roof    
                new_pos = (new_pos[0] - 1, new_pos[1])           # If not he keeps on going
                
                if Busted(enemies, new_pos, pMM):
                    if MM:
                        return True, new_pos, R
                    if Fans:
                        R = -(randint(20, 100))
    else:
        if rocket_motor and random() < .3:                       # Checks if his rocket motor is available
            R = -3                                               # There is only a 30% chance that the motor actually works
            if grid[new_pos[0] - 1, new_pos[1]] != 2:            # Checks if he will hit a roof                            
                new_pos = (new_pos[0] - 1, new_pos[1])           # The rocket motor boost him up and away!
        else:
            R = 0
            new_pos = (new_pos[0] + 1, new_pos[1])               # He falls down one step

            if Busted(enemies, new_pos, pMM):
                if MM:
                    return True, new_pos, R
                if Fans:
                    R = -(randint(20, 100))

            if grid[new_pos[0] + 1, new_pos[1]] not in solids:   # Checks if he hit the ground
                new_pos = (new_pos[0] + 1, new_pos[1])           # If not he falls further
                
                if Busted(enemies, new_pos, pMM):
                    if MM:
                        return True, new_pos, R
                    if Fans:
                        R = -(randint(20, 100))
    
    return False, new_pos, R
