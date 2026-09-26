"""
Shell visualization for the reel demo, receiving status updates from the cassette simulator.
"""
import pygame
from .reel_demo import ReelGraphics

class ReelDemoShell:
    def __init__(self, rect):
        self.reel_graphics = ReelGraphics(rect)

    def update_status(
        self,
        tape_position,
        motor_position,
        reading,
        writing,
        activity=None,
    ):
        previous_right = self.reel_graphics.right_tape
        self.reel_graphics.left_tape = tape_position[0]
        self.reel_graphics.right_tape = tape_position[1]
        self.reel_graphics.pending_motion += tape_position[1] - previous_right
        self.reel_graphics.motor_position = motor_position
        self.reel_graphics.left.tape_length = tape_position[0]
        self.reel_graphics.right.tape_length = tape_position[1]
        self.reel_graphics.position_driven = True
        self.reel_graphics.set_state(
            playing=reading,
            recording=writing,
            seeking=(activity == "seek"),
            seek_dir=(1 if self.reel_graphics.pending_motion >= 0 else -1),
        )

    def update(self, dt):
        self.reel_graphics.update(dt)

    def draw(self, surface):
        self.reel_graphics.draw(surface)
