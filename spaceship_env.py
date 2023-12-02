import pygame as pg
import gym
import numpy as np
from random import randrange

BACKGROUND_COLOR = (0, 30, 120)
ACCELERATION  = 0.03
DECELERATION = 0.03


class Env(gym.Env):
    def __init__(self):
        # just realized this never uses the super(), not a problem I guess?
        # self.action_space = gym.spaces.Discrete(3, start=-1)
        self.action_space = [-1, 0, 1]

        self.observation_space = gym.spaces.Box(
        low = -100,
        high = 1400,
        shape = (7,),
        dtype = 'uint8'
        )

        self.spaceship = Spaceship(np.array([100, 100], dtype=float), np.array([0, 0], dtype=float), 0, [25, 50])
        self.target = Target(np.array([500, 500]), np.array([0, 0]), (30, 30), (200, 30, 30))
        self.screen_dimension = (1400, 1000)
        self.ui = Ui(self.screen_dimension)

    def return_state(self):
        return np.array([self.spaceship.pos[0], self.spaceship.pos[1],
                          self.spaceship.velocity[0], self.spaceship.velocity[1], 
                          self.spaceship.rotation, self.target.pos[0], self.target.pos[1]])

    def step(self, action):
        first_distance = abs(self.spaceship.pos[0] - self.target.pos[0]) + abs(self.spaceship.pos[1] - self.target.pos[1])
        self.spaceship.rotation += 90 * action
        self.spaceship.rotation %= 360

        self.spaceship.pos += self.spaceship.velocity

        done = False
        reward = -0.1

        # if action == 0:
        #     reward += 0.1

        # if abs(self.spaceship.velocity[0]) < 1 and abs(self.spaceship.velocity[1]) < 1:
        #     reward -= 50
        # if abs(self.spaceship.velocity[0]) < 0.5 and abs(self.spaceship.velocity[1]) < 0.5:
        #     reward -= 500
        # distance = abs(self.spaceship.pos[0] - self.target.pos[0]) + abs(self.spaceship.pos[1] - self.target.pos[1])

        # if distance < first_distance:
        #     max_velocity = max(self.spaceship.velocity)
        #     if max_velocity > 0.5:
        #         reward += 15 * max(self.spaceship.velocity)
        # else:
        #     reward -= 100

        if self.check_for_collision(self.spaceship, self.target):
            reward += 1000
            done = True
            # print("good job!")

        if self.check_for_out_of_bounds(self.spaceship):
            reward -= 500
            done = True
            # print("BAD. Bad.")
        
        self.update(self.spaceship)

        state = self.return_state()
        # print("state:", state)
        return state, reward, done, None

    def update(self, object1):
        for i in range(object1.velocity.shape[0]):
            if object1.velocity[i] > 0:
                object1.velocity[i] = min(max(0, object1.velocity[i] - DECELERATION), 100)
            else:
                object1.velocity[i] = max(min(0, object1.velocity[i] + DECELERATION), -100)
        if object1.rotation == 0:
            object1.velocity[0] += 2 * ACCELERATION
        elif object1.rotation / 90 == 1:
            object1.velocity[1] -= 2 * ACCELERATION
        elif object1.rotation / 90 == 2:
            object1.velocity[0] -= 2 * ACCELERATION
        elif object1.rotation / 90 == 3:
            object1.velocity[1] += 2 * ACCELERATION

    def check_for_collision(self, object1, object2):
        if object1.rotation == 0 or object1.rotation == 180:
            sizes = [object1.size[1], object1.size[0]]
        else:
            sizes = [object1.size[0], object1.size[1]]
        # print(object1.pos[0] - object2.pos[0], width + object2.size[0])
        # print(object1.pos[1] - object2.pos[1], height + object2.size[1])
        # print()

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
    
    def render(self):
        self.ui.draw(self.spaceship, self.target)
        pg.display.update()

    def reset(self):
        print("reset")
        if self.check_for_out_of_bounds(self.spaceship):
            self.spaceship = Spaceship(np.array([100, 100], dtype=float), np.array([0, 0], dtype=float), 0, [25, 50])
        random_pos = [randrange(0, self.screen_dimension[0]), randrange(0, self.screen_dimension[1])]
        self.target = Target(np.array(random_pos), np.array([0, 0]), (30, 30), (200, 30, 30))
        return self.return_state()
    
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
        self.rotation = rotation
        self.image = pg.image.load("Images/spaceship_image.png")
        self.image = pg.transform.scale(self.image, size)

    def draw(self, screen):
        image = pg.transform.rotate(self.image, self.rotation - 90)
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
        self.screen = None

    def init_render(self):
        self.screen = pg.display.set_mode((self.screen_dimension), pg.RESIZABLE)
        self.screen.fill(BACKGROUND_COLOR)

    def draw(self, spaceship, target):
        self.screen.fill(BACKGROUND_COLOR)
        spaceship.draw(self.screen)
        target.draw(self.screen)

    
if __name__ == "__main__":

    env = Env()

    env.ui.init_render()
    env.render()
    # pause = input("pause")
    clock = pg.time.Clock()

    while True:
        clock.tick(150)
        action = 0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_a:
                    action = 1
                elif event.key == pg.K_d:
                    action = -1
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
            





