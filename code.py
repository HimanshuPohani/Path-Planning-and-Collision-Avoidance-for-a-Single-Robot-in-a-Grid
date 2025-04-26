import numpy as np
import matplotlib.pyplot as plt
import heapq


def create_grid(rows=20, cols=20, obstacle_ratio=0.2):
    grid = np.zeros((rows, cols))
    num_obstacles = int(rows * cols * obstacle_ratio)
    for _ in range(num_obstacles):
        x, y = np.random.randint(0, rows), np.random.randint(0, cols)
        grid[x][y] = 1  
    return grid


def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                heapq.heappush(open_set, (cost+1 + heuristic((nx, ny), goal), cost+1, (nx, ny), path + [(nx, ny)]))
    return []


def plot_path(grid, path, start, goal):
    plt.imshow(grid, cmap='Greys')
    if path:
        x, y = zip(*path)
        plt.plot(y, x, color='blue', linewidth=2, label="Path")
    plt.scatter(start[1], start[0], c='green', s=100, label="Start")
    plt.scatter(goal[1], goal[0], c='red', s=100, label="Goal")
    plt.legend()
    plt.title("Path Planning and Collision Avoidance")
    plt.show()


if __name__ == "__main__":
    grid = create_grid()
    start = (0, 0)
    goal = (19, 19)
    
    
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    path = a_star(grid, start, goal)
    if path:
        print("Path found!")
    else:
        print("No path found.")
    
    plot_path(grid, path, start, goal)
