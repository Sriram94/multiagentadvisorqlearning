import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        
        hell2_center = origin + np.array([UNIT*2, UNIT * 3])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        self.rect1 = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        self.rect2 = self.canvas.create_rectangle(
            165.0, 165.0,
            195.0, 195.0,
            fill='blue')
        
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)
        origin = np.array([20, 20])
        self.rect1 = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        self.rect2 = self.canvas.create_rectangle(
            165.0, 165.0,
            195.0, 195.0,
            fill='blue')
        
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2)

    def step(self, action1, action2):
        s = self.canvas.coords(self.rect1)
        s2 = self.canvas.coords(self.rect2)
        base_action1 = np.array([0, 0])
        base_action2 = np.array([0, 0])
        
        if action1 == 0:   # up
            if s[1] > UNIT:
                base_action1[1] -= UNIT
        elif action1 == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
        elif action1 == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
        elif action1 == 3:   # left
            if s[0] > UNIT:
                base_action1[0] -= UNIT

        if action2 == 0:   # up
            if s2[1] > UNIT:
                base_action2[1] -= UNIT
        elif action2 == 1:   # down
            if s2[1] < (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action2 == 2:   # right
            if s2[0] < (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action2 == 3:   # left
            if s2[0] > UNIT:
                base_action2[0] -= UNIT
        self.canvas.move(self.rect1, base_action1[0], base_action1[1])  # move agent
        self.canvas.move(self.rect2, base_action2[0], base_action2[1])  # move agent

        s_ = self.canvas.coords(self.rect1)  # next state
        s2_ = self.canvas.coords(self.rect2)  # next state
        reward = 0
        reward2 = 0
        done = False
        
        
        if s_ == self.canvas.coords(self.oval) and s2_ == self.canvas.coords(self.oval): 
            reward = 2
            reward2 = 2
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'

        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)] and s2_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -2 
            reward2 = -2
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'


        elif s_ == self.canvas.coords(self.oval) and s2_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = 1
            reward2 = 1
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'

        elif s2_ == self.canvas.coords(self.oval) and s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = 1
            reward2 = 1
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'
        
        elif s_ == self.canvas.coords(self.oval):
            reward = 1
            reward2 = 1
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'
        
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            reward2 = -1
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'


        elif s2_ == self.canvas.coords(self.oval):
            reward2 = 1
            reward = 1
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'
        
        elif s2_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            reward2 = -1
            done = True
            s_ = 'terminal'
            s2_ = 'terminal'

        
        return s_, s2_, reward, reward2, done

    
    def render(self):
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
