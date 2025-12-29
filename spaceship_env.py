import pygame as pg
import gymnasium as gym
import numpy as np
from random import randrange

BACKGROUND_COLOR = (0, 30, 120)
ACCELERATION  = 0.2
DECELERATION = 0.04
MAX_V = 200
DIRECTIONS = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


class SpaceshipEnv(gym.Env):
    '''
        3 actions are left, no turn, and right
        Observation space is pos, vel, target_pos
        '''
    def __init__(self, headless=True, return_pixels=False):
        self.headless = headless
        self.return_pixels = return_pixels
        self.screen_dimension = (1400, 1000)
        self.action_space = gym.spaces.Discrete(3)   # 0 is left, 1 is no turn, 2 is right
        
        self.step_count = 0

        self.observation_space = gym.spaces.Dict({
            "position" : gym.spaces.Box(low=np.array([0, 0]), high=np.array(self.screen_dimension), shape=(2,)),
            "velocity" : gym.spaces.Box(low=0, high=MAX_V, shape=(2,)),
            "rotation" : gym.spaces.Box(low=0, high=1, shape=(4,)),
            "step_count" : gym.spaces.Box(low=0, high=5000, shape=(1,), dtype=float)  # actually not sure what the upper bound on this is
        })
        
        if self.return_pixels:
            self.observation_space['pixels'] = gym.spaces.Box(low=0, high=255, shape=(self.screen_dimension[0], self.screen_dimension[1], 3), dtype=np.uint8)

        self.spaceship = Spaceship(np.array([100, 100], dtype=float), np.array([0, 0], dtype=float), 0, [25, 50])
        self.target = Target(np.array([500, 500]), np.array([0, 0]), (30, 30), (200, 30, 30))
        self.ui = Ui(self.screen_dimension)

    def return_state(self):
        # print("actually returning", DIRECTIONS[self.spaceship.dir_index])
        state = {
            "position" : np.array(self.spaceship.pos),
            "velocity" : np.array(self.spaceship.velocity),
            "rotation" : DIRECTIONS[self.spaceship.dir_index],
            "step_count" : self.step_count,
        }
        if self.return_pixels:
            state['pixels'] = self.render()
        return state

    def step(self, action):
        self.spaceship.update(action)
        # print("updated?", self.spaceship.dir_index)

        done = False
        reward = -0.02

        if self.check_for_collision(self.spaceship, self.target):
            reward += 1
            done = True
            # print("good job!")

        if self.check_for_out_of_bounds(self.spaceship):
            reward -= 0.5
            done = True
        
        self.step_count += 1

        state = self.return_state()
        return state, reward, done, False, None

    def check_for_collision(self, object1, object2):
        if object1.dir_index == 0 or object1.dir_index == 2:
            sizes = [object1.size[1], object1.size[0]]
        else:
            sizes = [object1.size[0], object1.size[1]]

        dists = [object1.pos[0] - object2.pos[0], object1.pos[1] - object2.pos[1]]
        for i, dist in enumerate(dists):
            if dist < 0:
                if dist < -(sizes[i] + object2.size[i]):
                    return False
            else:
                if dist > object2.size[i]:
                    return False
        return True
    
    def check_for_out_of_bounds(self, object1):
        if object1.pos[0] < 0:
            return True
        if object1.pos[1] < 0:
            return True
        if object1.pos[0] + object1.size[0] > self.screen_dimension[0]:
            return True
        if object1.pos[1] + object1.size[1] > self.screen_dimension[1]:
            return True
        return False
    
    def render(self, render_mode="rgb_array"):
        self.ui.draw(self.spaceship, self.target)
        if not self.headless:
            pg.display.update()
        return pg.surfarray.array3d(self.ui.screen)

    def reset(self, seed=None, options=None):
        if self.check_for_out_of_bounds(self.spaceship):
            self.spaceship = Spaceship(np.array([100, 100], dtype=float), np.array([0, 0], dtype=float), 0, [25, 50])
        random_pos = [randrange(0, self.screen_dimension[0]), randrange(0, self.screen_dimension[1])]
        self.target = Target(np.array(random_pos), np.array([0, 0]), (30, 30), (200, 30, 30))
        self.step_count = 0
        return self.return_state(), {}
    
class Body():
    def __init__(self, pos, velocity, size):
        self.pos = pos
        self.velocity = velocity
        self.size = size
        self.image = pg.image.load("Images/spaceship_image.png")
        self.image = pg.transform.scale(self.image, size)

    def draw(self):
        return

class Spaceship(Body):
    def __init__(self, pos, velocity, rotation, size):
        super().__init__(pos, velocity, size)
        self.dir_index = rotation
        self.image = pg.image.load("Images/spaceship_image.png")
        self.image = pg.transform.scale(self.image, size)
        
    def update(self, action):
        if action == 0:
            self.dir_index = (self.dir_index - 1) % 4
        elif action == 2:
            self.dir_index = (self.dir_index + 1) % 4
        
        if self.dir_index == 0:
            self.velocity[0] += ACCELERATION
        elif self.dir_index == 1:
            self.velocity[1] -= ACCELERATION
        elif self.dir_index == 2:
            self.velocity[0] -= ACCELERATION
        elif self.dir_index == 3:
            self.velocity[1] += ACCELERATION
        else:
            print("dir_index not recognized:", self.dir_index)
            
        for i in range(len(self.velocity)):
            if self.velocity[i] > 0:
                self.velocity[i] = max(0, self.velocity[i] - DECELERATION)
            else:
                self.velocity[i] = min(0, self.velocity[i] + DECELERATION)

        self.velocity = np.clip(self.velocity, -MAX_V, MAX_V)
        
        print(self.velocity)
        
        self.pos += self.velocity
        
        
    def draw(self, screen):
        image = pg.transform.rotate(self.image, (self.dir_index - 1) * 90)
        screen.blit(image, self.pos)

class Target(Body):
    def __init__(self, pos, velocity, size, color):
        super().__init__(pos, velocity, size)
        self.color = color

    def draw(self, screen):
        pg.draw.circle(screen, self.color, self.pos, self.size[0])

class Ui():
    def __init__(self, screen_dimension):
        self.screen_dimension = screen_dimension
        self.screen: pg.surface = None
        self.init_render()

    def init_render(self):
        self.screen = pg.display.set_mode((self.screen_dimension), pg.RESIZABLE)
        self.screen.fill(BACKGROUND_COLOR)

    def draw(self, spaceship, target):
        self.screen.fill(BACKGROUND_COLOR)
        spaceship.draw(self.screen)
        target.draw(self.screen)

    
if __name__ == "__main__":

    env = SpaceshipEnv(headless=False)

    env.ui.init_render()
    env.render()
    # pause = input("pause")
    clock = pg.time.Clock()

    while True:
        clock.tick(30)
        action = 1
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_a:
                    action = 2
                elif event.key == pg.K_d:
                    action = 0
                elif event.key == pg.K_p:
                    unpause = False
                    while True:
                        if unpause:
                            break
                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                pg.quit()
                            if event.type == pg.KEYDOWN:
                                print("keydown")
                                if event.key == pg.K_p:
                                    unpause = True
                                    break
                
        # if pg.key.get_pressed()[pg.K_a]:
        #     env.step(-1)
        # elif pg.key.get_pressed()[pg.K_d]:
        #     env.step(1)
        # else:
        #     env.step(0)
        state, reward, done, _, _ = env.step(action)
        if done:
            env.reset()
            done = False
        env.render()
            





