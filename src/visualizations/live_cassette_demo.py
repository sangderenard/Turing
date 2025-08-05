"""Run a full cassette demo with live audio and reel animation.

This script compiles a simple program, writes it to a virtual cassette, and
then executes it while displaying a real-time visualization of the tape.
"""
from __future__ import annotations

import threading
import time

import pygame

from ..compiler.bitops_translator import BitOpsTranslator
from ..compiler.tape_compiler import TapeCompiler
from ..hardware.cassette_tape import CassetteTapeBackend
from ..turing_machine.survival_computer import prime_tape_with_program
from ..turing_machine.tape_machine import TapeMachine

from .reel_demo_shell import ReelDemoShell

BACKGROUND = (30, 30, 30)


def run_with_visual(func, shell: ReelDemoShell, screen, *args) -> bool:
    """Run ``func`` in a thread while updating the reel visualization."""
    thread = threading.Thread(target=func, args=args)
    thread.start()
    clock = pygame.time.Clock()
    running = True
    while running and thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        dt = clock.tick(60) / 1000.0
        shell.update(dt)
        screen.fill(BACKGROUND)
        shell.draw(screen)
        pygame.display.flip()
    thread.join()
    return running


def main() -> None:
    BIT_WIDTH = 32
    TAPE_LEN = 8192

    translator = BitOpsTranslator(bit_width=BIT_WIDTH)
    translator.bit_mul(5, 3)
    compiler = TapeCompiler(translator.graph, BIT_WIDTH)
    tape_map, instructions = compiler.compile()
    frames = TapeCompiler.binarize_instructions(instructions)

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    shell = ReelDemoShell(screen.get_rect())

    cassette = CassetteTapeBackend(tape_length=TAPE_LEN, status_callback=shell.update_status)
    shell.reel_graphics.total_tape = cassette.total_bits

    if not run_with_visual(prime_tape_with_program, shell, screen, cassette, tape_map, frames):
        pygame.quit()
        return

    machine = TapeMachine(cassette, BIT_WIDTH)
    if not run_with_visual(machine.run, shell, screen, len(instructions)):
        pygame.quit()
        return

    time.sleep(1.0)
    cassette.close()
    pygame.quit()


if __name__ == "__main__":
    main()

