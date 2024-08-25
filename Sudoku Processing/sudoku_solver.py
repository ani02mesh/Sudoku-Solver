import numpy as np

def is_valid(grid,r,c,k):
    if k not in grid[:,c] and k not in grid[r]:
        aa = [grid[i][j] for i in range((r//3)*3,((r//3)*3)+3) for j in range((c//3)*3,((c//3)*3)+3)]
        return True if k not in aa else False
    else:
        return False

def solver(grid,r,c):
    if r == 9:
        return True
    elif c==9:
        return solver(grid,r+1,0)
    elif grid[r][c] != 0 :
        return solver(grid,r,c+1)
    else:
        for k in range(1,10):
            if is_valid(grid,r,c,k):
                grid[r][c] = k
                if solver(grid,r,c+1):
                    return True
                grid[r][c] = 0
        return False