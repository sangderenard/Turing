"""Pygame demo of spinning tape reels for cassette or reel-to-reel machines."""

from __future__ import annotations

import argparse
import math
import sys
import pygame

from reel_math import tape_radius, tangent_points

BROWN = (150, 75, 0)
GREY = (220, 220, 220)
DARK_GREY = (50, 50, 50)
BACKGROUND = (30, 30, 30)
GREEN_ON = (0, 255, 0)
GREEN_OFF = (0, 120, 0)
RED_ON = (255, 0, 0)
RED_OFF = (120, 0, 0)


class Reel:
    def __init__(self, center: tuple[int, int], spool_radius: float, max_radius: float, tape_length: float):
        self.center = center
        self.spool_radius = spool_radius
        self.max_radius = max_radius
        self.tape_length = tape_length
        self.angle = 0.0

    def update(self, dt: float, linear_speed: float, total_tape: float):
        if self.tape_length <= 0:
            radius = self.spool_radius
        else:
            radius = tape_radius(self.spool_radius, self.max_radius, self.tape_length, total_tape)
        # angular velocity = v / r
        if linear_speed != 0:
            self.angle = (self.angle + linear_speed / max(radius, 1) * dt) % (2 * math.pi)
        return radius

    def draw(self, surface: pygame.Surface, radius: float):
        pygame.draw.circle(surface, BROWN, self.center, int(radius))
        pygame.draw.circle(surface, GREY, self.center, int(self.spool_radius))
        # draw spokes
        for i in range(2):
            phi = self.angle + i * math.pi / 2
            x1 = self.center[0] + self.spool_radius * math.cos(phi)
            y1 = self.center[1] + self.spool_radius * math.sin(phi)
            x2 = self.center[0] - self.spool_radius * math.cos(phi)
            y2 = self.center[1] - self.spool_radius * math.sin(phi)
            pygame.draw.line(surface, DARK_GREY, (x1, y1), (x2, y2), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cassette", "reel"], default="cassette", help="Display cassette or reel-to-reel")
    args = parser.parse_args()

    if args.mode == "cassette":
        spool_radius = 40
        max_radius = 80
        tape_speed = 60.0
        left_center = (250, 300)
        right_center = (550, 300)
    else:
        spool_radius = 60
        max_radius = 150
        tape_speed = 120.0
        left_center = (200, 300)
        right_center = (600, 300)

    head_pos = (400, 260)
    total_tape = 1000.0
    left_tape = total_tape
    right_tape = 0.0
    playing = True
    recording = False

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    left = Reel(left_center, spool_radius, max_radius, left_tape)
    right = Reel(right_center, spool_radius, max_radius, right_tape)

    while True:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 0
                if event.key == pygame.K_SPACE:
                    playing = not playing
                if event.key == pygame.K_w:
                    recording = not recording

        if playing and left_tape > 0:
            moved = tape_speed * dt
            left_tape = max(0.0, left_tape - moved)
            right_tape = min(total_tape, right_tape + moved)
        else:
            moved = 0.0

        left.tape_length = left_tape
        right.tape_length = right_tape
        left_radius = left.update(dt, tape_speed if playing else 0.0, total_tape)
        right_radius = right.update(dt, tape_speed if playing else 0.0, total_tape)

        screen.fill(BACKGROUND)

        # compute tangent points for tape path
        lt1, lt2 = tangent_points(*left.center, left_radius, *head_pos)
        rt1, rt2 = tangent_points(*right.center, right_radius, *head_pos)
        left_tangent = lt1 if lt1[1] < lt2[1] else lt2
        right_tangent = rt1 if rt1[1] < rt2[1] else rt2
        pygame.draw.line(screen, BROWN, left_tangent, head_pos, 6)
        pygame.draw.line(screen, BROWN, head_pos, right_tangent, 6)

        left.draw(screen, left_radius)
        right.draw(screen, right_radius)

        # draw head
        head_rect = pygame.Rect(0, 0, 30, 40)
        head_rect.center = head_pos
        pygame.draw.rect(screen, GREY, head_rect)
        read_color = GREEN_ON if playing else GREEN_OFF
        write_color = RED_ON if recording and playing else RED_OFF
        read_rect = pygame.Rect(head_pos[0] - 10, head_pos[1] - 30, 10, 15)
        write_rect = pygame.Rect(head_pos[0], head_pos[1] - 30, 10, 15)
        pygame.draw.rect(screen, read_color, read_rect)
        pygame.draw.rect(screen, write_color, write_rect)

        pygame.display.flip()

    return 0


if __name__ == "__main__":
    sys.exit(main())

