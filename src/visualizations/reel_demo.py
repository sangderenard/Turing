# Move reel_demo.py to visualizations folder for organization
# This is the same code as before, just relocated.

"""Composable ReelGraphics class for Pygame UI integration."""

import math
import pygame
from ..reel_math import tape_radius, tangent_points

BROWN = (150, 75, 0)
GREY = (220, 220, 220)
DARK_GREY = (50, 50, 50)
BACKGROUND = (30, 30, 30)
GREEN_ON = (0, 255, 0)
GREEN_OFF = (0, 120, 0)
RED_ON = (255, 0, 0)
RED_OFF = (120, 0, 0)

class Reel:
    def __init__(self, center, spool_radius, max_radius, tape_length):
        self.center = center
        self.spool_radius = spool_radius
        self.max_radius = max_radius
        self.tape_length = tape_length
        self.angle = 0.0

    def update(self, dt, linear_speed, total_tape):
        if self.tape_length <= 0:
            radius = self.spool_radius
        else:
            radius = tape_radius(self.spool_radius, self.max_radius, self.tape_length, total_tape)
        if linear_speed != 0:
            self.angle = (self.angle + linear_speed / max(radius, 1) * dt) % (2 * math.pi)
        return radius

    def draw(self, surface, radius):
        pygame.draw.circle(surface, BROWN, self.center, int(radius))
        pygame.draw.circle(surface, GREY, self.center, int(self.spool_radius))
        for i in range(2):
            phi = self.angle + i * math.pi / 2
            x1 = self.center[0] + self.spool_radius * math.cos(phi)
            y1 = self.center[1] + self.spool_radius * math.sin(phi)
            x2 = self.center[0] - self.spool_radius * math.cos(phi)
            y2 = self.center[1] - self.spool_radius * math.sin(phi)
            pygame.draw.line(surface, DARK_GREY, (x1, y1), (x2, y2), 4)

class ReelGraphics:
    def __init__(self, rect, mode="cassette"):
        self.rect = rect
        if mode == "cassette":
            self.spool_radius = 40
            self.max_radius = 80
            self.tape_speed = 60.0
            self.seek_speed = 200.0
            self.left_center = (rect.left + 70, rect.centery)
            self.right_center = (rect.right - 70, rect.centery)
        else:
            self.spool_radius = 60
            self.max_radius = 150
            self.tape_speed = 120.0
            self.seek_speed = 300.0
            self.left_center = (rect.left + 100, rect.centery)
            self.right_center = (rect.right - 100, rect.centery)
        self.head_pos = (rect.centerx, rect.centery - 40)
        self.total_tape = 1000.0
        self.left_tape = self.total_tape
        self.right_tape = 0.0
        self.playing = False
        self.recording = False
        self.seeking = False
        self.seek_dir = 1
        self.head_gap = 30
        self.head_anim = 0.0
        self.head_anim_speed = 4.0
        self.left = Reel(self.left_center, self.spool_radius, self.max_radius, self.left_tape)
        self.right = Reel(self.right_center, self.spool_radius, self.max_radius, self.right_tape)

    def set_state(self, playing=None, recording=None, seeking=None, seek_dir=None):
        if playing is not None:
            self.playing = playing
        if recording is not None:
            self.recording = recording
        if seeking is not None:
            self.seeking = seeking
        if seek_dir is not None:
            self.seek_dir = seek_dir

    def update(self, dt):
        moved = 0.0
        if self.playing and self.left_tape > 0:
            moved = self.tape_speed * dt
            self.left_tape = max(0.0, self.left_tape - moved)
            self.right_tape = min(self.total_tape, self.right_tape + moved)
        elif self.recording and self.left_tape > 0:
            moved = self.tape_speed * dt
            self.left_tape = max(0.0, self.left_tape - moved)
            self.right_tape = min(self.total_tape, self.right_tape + moved)
        elif self.seeking:
            moved = self.seek_speed * dt * self.seek_dir
            self.left_tape = min(max(self.left_tape - moved, 0.0), self.total_tape)
            self.right_tape = self.total_tape - self.left_tape
        self.left.tape_length = self.left_tape
        self.right.tape_length = self.right_tape
        self.left_radius = self.left.update(dt, moved if (self.playing or self.recording or self.seeking) else 0.0, self.total_tape)
        self.right_radius = self.right.update(dt, moved if (self.playing or self.recording or self.seeking) else 0.0, self.total_tape)
        # Head animation
        head_down = self.playing or self.recording
        if head_down:
            self.head_anim = min(self.head_anim + self.head_anim_speed * dt, 1.0)
        else:
            self.head_anim = max(self.head_anim - self.head_anim_speed * dt, 0.0)

    def draw(self, surface):
        # Tape path
        lt1, lt2 = tangent_points(*self.left.center, self.left_radius, *self.head_pos)
        rt1, rt2 = tangent_points(*self.right.center, self.right_radius, *self.head_pos)
        left_tangent = lt1 if lt1[1] < lt2[1] else lt2
        right_tangent = rt1 if rt1[1] < rt2[1] else rt2
        pygame.draw.line(surface, BROWN, left_tangent, self.head_pos, 6)
        pygame.draw.line(surface, BROWN, self.head_pos, right_tangent, 6)
        self.left.draw(surface, self.left_radius)
        self.right.draw(surface, self.right_radius)
        # Head position (animated)
        head_y = self.head_pos[1] - self.head_gap * (1.0 - self.head_anim)
        head_rect = pygame.Rect(0, 0, 30, 40)
        head_rect.center = (self.head_pos[0], head_y)
        pygame.draw.rect(surface, GREY, head_rect)
        read_color = GREEN_ON if self.playing else GREEN_OFF
        write_color = RED_ON if self.recording else RED_OFF
        read_rect = pygame.Rect(self.head_pos[0] - 10, head_y - 30, 10, 15)
        write_rect = pygame.Rect(self.head_pos[0], head_y - 30, 10, 15)
        pygame.draw.rect(surface, read_color, read_rect)
        pygame.draw.rect(surface, write_color, write_rect)
        # Status text
        font = pygame.font.SysFont(None, 24)
        status = "PLAY" if self.playing else "REC" if self.recording else "SEEK" if self.seeking else "STOP"
        text = font.render(f"{status} | Tape: {self.left_tape:.1f} / {self.total_tape}", True, (255,255,255))
        surface.blit(text, (self.rect.left + 10, self.rect.top + 10))
