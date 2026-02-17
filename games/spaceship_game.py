#!/usr/bin/env python3
"""Two-player spaceship game: Human vs PPO-trained AI."""

import sys
import os
import argparse

import numpy as np
import pygame as pg
import torch
from torch import nn
from random import randrange

# ---------------------------------------------------------------------------
# Constants (matching spaceship_env.py)
# ---------------------------------------------------------------------------
BACKGROUND_COLOR = (0, 30, 120)
ACCELERATION = 0.2
DECELERATION = 0.04
MAX_V = 200
SCREEN_W, SCREEN_H = 1400, 1000
TURN_COOLDOWN = 10
TARGET_RADIUS = 30
WIN_SCORE = 3
FPS = 30

DIRECTIONS = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

SHIP_SIZE = (25, 50)

# Resolve path to the sprite image
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_PATH = os.path.join(_SCRIPT_DIR, "..", "spaceship_learn", "Images", "spaceship_image.png")
_CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "..", "spaceship_learn", "checkpoints")

# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------

class Spaceship:
    def __init__(self, pos, dir_index, image, tint=None):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.dir_index = dir_index
        self.time_since_turn = TURN_COOLDOWN

        self.base_image = pg.transform.scale(image, SHIP_SIZE)
        if tint is not None:
            self.base_image = self._tint(self.base_image, tint)

    @staticmethod
    def _tint(surface, color):
        tinted = surface.copy()
        tinted.fill(color, special_flags=pg.BLEND_MULT)
        return tinted

    def update(self, action):
        # action: 0=left, 1=none, 2=right
        if self.time_since_turn >= TURN_COOLDOWN:
            if action == 0:
                self.dir_index = (self.dir_index - 1) % 4
                self.time_since_turn = -1
            elif action == 2:
                self.dir_index = (self.dir_index + 1) % 4
                self.time_since_turn = -1

        self.time_since_turn = min(self.time_since_turn + 1, TURN_COOLDOWN)

        if self.dir_index == 0:
            self.velocity[0] += ACCELERATION
        elif self.dir_index == 1:
            self.velocity[1] -= ACCELERATION
        elif self.dir_index == 2:
            self.velocity[0] -= ACCELERATION
        elif self.dir_index == 3:
            self.velocity[1] += ACCELERATION

        for i in range(2):
            if self.velocity[i] > 0:
                self.velocity[i] = max(0, self.velocity[i] - DECELERATION)
            else:
                self.velocity[i] = min(0, self.velocity[i] + DECELERATION)

        self.velocity = np.clip(self.velocity, -MAX_V, MAX_V)
        self.pos += self.velocity

    def draw(self, screen):
        image = pg.transform.rotate(self.base_image, (self.dir_index - 1) * 90)
        screen.blit(image, self.pos)

    @property
    def size(self):
        return SHIP_SIZE


class Target:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.radius = TARGET_RADIUS

    def draw(self, screen):
        pg.draw.circle(screen, (200, 30, 30), self.pos.astype(int), self.radius)


# ---------------------------------------------------------------------------
# AI controller
# ---------------------------------------------------------------------------

class AIController:
    def __init__(self, checkpoint_path):
        self.actor = nn.Sequential(
            nn.Linear(10, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, ship, target_pos):
        obs = self._build_obs(ship, target_pos)
        logits = self.actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(logits.argmax(dim=-1).item())

    @staticmethod
    def _build_obs(ship, target_pos):
        pos_norm = np.array([ship.pos[0] / SCREEN_W, ship.pos[1] / SCREEN_H], dtype=np.float32)
        tgt_norm = np.array([target_pos[0] / SCREEN_W, target_pos[1] / SCREEN_H], dtype=np.float32)
        vel_norm = np.array([ship.velocity[0] / MAX_V, ship.velocity[1] / MAX_V], dtype=np.float32)
        rot = DIRECTIONS[ship.dir_index]
        return np.concatenate([pos_norm, tgt_norm, vel_norm, rot])


# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------

def check_ship_target_collision(ship, target):
    """Check if ship's bounding box overlaps the target circle."""
    if ship.dir_index in (0, 2):
        w, h = SHIP_SIZE[1], SHIP_SIZE[0]
    else:
        w, h = SHIP_SIZE[0], SHIP_SIZE[1]
    # Closest point on the rectangle to the circle centre
    cx = np.clip(target.pos[0], ship.pos[0], ship.pos[0] + w)
    cy = np.clip(target.pos[1], ship.pos[1], ship.pos[1] + h)
    dist_sq = (target.pos[0] - cx) ** 2 + (target.pos[1] - cy) ** 2
    return dist_sq <= target.radius ** 2


def check_out_of_bounds(ship):
    if ship.dir_index in (0, 2):
        w, h = SHIP_SIZE[1], SHIP_SIZE[0]
    else:
        w, h = SHIP_SIZE[0], SHIP_SIZE[1]
    if ship.pos[0] < 0 or ship.pos[1] < 0:
        return True
    if ship.pos[0] + w > SCREEN_W or ship.pos[1] + h > SCREEN_H:
        return True
    return False


# ---------------------------------------------------------------------------
# Main game
# ---------------------------------------------------------------------------

class Game:
    def __init__(self, checkpoint_path):
        pg.init()
        self.screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
        pg.display.set_caption("Spaceship Game — Human vs AI")
        self.clock = pg.time.Clock()

        self.ship_image = pg.image.load(_IMAGE_PATH).convert_alpha()
        self.ai = AIController(checkpoint_path)

        self.font_large = pg.font.SysFont("Arial", 48, bold=True)
        self.font_medium = pg.font.SysFont("Arial", 36)
        self.font_small = pg.font.SysFont("Arial", 24)

        self.reset_game()

    # -- state management ---------------------------------------------------

    def reset_game(self):
        self.player = Spaceship([200, 500], 0, self.ship_image)
        self.ai_ship = Spaceship([1200, 500], 2, self.ship_image, tint=(255, 100, 100))
        self.target = self._random_target()
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.winner = None
        self.paused = False

    def _random_target(self):
        margin = 80
        while True:
            x = randrange(margin, SCREEN_W - margin)
            y = randrange(margin, SCREEN_H - margin)
            pos = np.array([x, y], dtype=float)
            # Ensure target isn't too close to either ship
            if np.linalg.norm(pos - self.player.pos) > 150 and np.linalg.norm(pos - self.ai_ship.pos) > 150:
                return Target(pos)

    def _respawn_ship(self, ship, start_pos, start_dir):
        ship.pos = np.array(start_pos, dtype=float)
        ship.velocity = np.array([0.0, 0.0])
        ship.dir_index = start_dir
        ship.time_since_turn = TURN_COOLDOWN

    # -- main loop ----------------------------------------------------------

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            # --- events ---
            player_action = 1  # default no-turn
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    break
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                        break
                    if self.game_over:
                        if event.key == pg.K_SPACE:
                            self.reset_game()
                        continue
                    if event.key == pg.K_p:
                        self.paused = not self.paused
                    if event.key in (pg.K_a, pg.K_LEFT):
                        player_action = 0
                    elif event.key in (pg.K_d, pg.K_RIGHT):
                        player_action = 2

            if not running:
                break

            # --- update ---
            if not self.game_over and not self.paused:
                self.player.update(player_action)
                ai_action = self.ai.get_action(self.ai_ship, self.target.pos)
                self.ai_ship.update(ai_action)
                self._check_events()

            # --- draw ---
            self._draw()

        pg.quit()

    def _check_events(self):
        # Player hits target
        if check_ship_target_collision(self.player, self.target):
            self.player_score += 1
            if self.player_score >= WIN_SCORE:
                self.game_over = True
                self.winner = "Player"
            else:
                self.target = self._random_target()

        # AI hits target
        if not self.game_over and check_ship_target_collision(self.ai_ship, self.target):
            self.ai_score += 1
            if self.ai_score >= WIN_SCORE:
                self.game_over = True
                self.winner = "AI"
            else:
                self.target = self._random_target()

        # Player hits wall
        if check_out_of_bounds(self.player):
            self.player_score -= 1
            self._respawn_ship(self.player, [200, 500], 0)

        # AI hits wall
        if check_out_of_bounds(self.ai_ship):
            self.ai_score -= 1
            self._respawn_ship(self.ai_ship, [1200, 500], 2)

    # -- rendering ----------------------------------------------------------

    def _draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.target.draw(self.screen)
        self.player.draw(self.screen)
        self.ai_ship.draw(self.screen)
        self._draw_hud()

        if self.game_over:
            self._draw_overlay()
        elif self.paused:
            self._draw_pause()

        pg.display.flip()

    def _draw_hud(self):
        score_text = f"Player: {self.player_score}  |  AI: {self.ai_score}"
        surf = self.font_large.render(score_text, True, (255, 255, 255))
        rect = surf.get_rect(midtop=(SCREEN_W // 2, 10))
        self.screen.blit(surf, rect)

        controls = "A/Left: turn left   D/Right: turn right   P: pause   ESC: quit"
        ctrl_surf = self.font_small.render(controls, True, (180, 180, 180))
        ctrl_rect = ctrl_surf.get_rect(midbottom=(SCREEN_W // 2, SCREEN_H - 10))
        self.screen.blit(ctrl_surf, ctrl_rect)

    def _draw_overlay(self):
        overlay = pg.Surface((SCREEN_W, SCREEN_H), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        win_text = f"{self.winner} Wins!"
        surf = self.font_large.render(win_text, True, (255, 255, 0))
        rect = surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 30))
        self.screen.blit(surf, rect)

        restart = "Press SPACE to play again"
        r_surf = self.font_medium.render(restart, True, (255, 255, 255))
        r_rect = r_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 30))
        self.screen.blit(r_surf, r_rect)

    def _draw_pause(self):
        overlay = pg.Surface((SCREEN_W, SCREEN_H), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))

        surf = self.font_large.render("PAUSED", True, (255, 255, 255))
        rect = surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2))
        self.screen.blit(surf, rect)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spaceship Game — Human vs AI")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt checkpoint. Defaults to the latest in spaceship_learn/checkpoints/",
    )
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Auto-pick latest checkpoint
        files = [f for f in os.listdir(_CHECKPOINT_DIR) if f.endswith(".pt")]
        if not files:
            print("No checkpoints found in", _CHECKPOINT_DIR)
            sys.exit(1)
        files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
        ckpt_path = os.path.join(_CHECKPOINT_DIR, files[-1])
        print(f"Using checkpoint: {ckpt_path}")

    game = Game(ckpt_path)
    game.run()


if __name__ == "__main__":
    main()
