"""Watch the physical spill specimen move the simulated cassette reels."""

from __future__ import annotations

import threading
import time

import networkx as nx
import pygame

from ..compiler.recursive_reduction import (
    assemble_nand_terminal_tape_program,
    execute_terminal_tape_program,
)
from .reel_demo_shell import ReelDemoShell


BACKGROUND = (30, 30, 30)


def _spill_specimen():
    graph = nx.MultiDiGraph(bit_width=4)
    for node, label in enumerate(("a", "b", "c", "d")):
        graph.add_node(node, op="input", label=label)
    for node in (4, 5, 6):
        graph.add_node(node, op="nand", label="nand")
    graph.add_edge(0, 4, arg_pos=0)
    graph.add_edge(1, 4, arg_pos=1)
    graph.add_edge(2, 5, arg_pos=0)
    graph.add_edge(3, 5, arg_pos=1)
    graph.add_edge(4, 6, arg_pos=0)
    graph.add_edge(5, 6, arg_pos=1)
    values = {0: 0b1100, 1: 0b1010, 2: 0b1111, 3: 0b0011}
    return assemble_nand_terminal_tape_program(
        graph,
        bit_width=4,
        input_values=values,
        output_nodes=(6,),
    )


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((720, 480))
    pygame.display.set_caption("Recursive reduction cassette — spill specimen")
    shell = ReelDemoShell(screen.get_rect())
    program = _spill_specimen()
    result = {}
    failure = {}

    def execute() -> None:
        try:
            result["witness"] = execute_terminal_tape_program(
                program,
                activity_callback=shell.update_status,
                time_scale_factor=0.002,
                play_audio=False,
            )
        except BaseException as exc:  # surface worker failures in the UI owner
            failure["exception"] = exc

    worker = threading.Thread(target=execute, name="cassette-spill-demo")
    worker.start()
    clock = pygame.time.Clock()
    running = True
    completed_at = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not worker.is_alive() and completed_at is None:
            completed_at = time.monotonic()
        if completed_at is not None and time.monotonic() - completed_at > 2.0:
            break
        dt = clock.tick(60) / 1000.0
        shell.update(dt)
        screen.fill(BACKGROUND)
        shell.draw(screen)
        pygame.display.flip()
    worker.join()
    pygame.quit()
    if failure:
        raise failure["exception"]
    witness = result.get("witness")
    if witness is not None:
        print(
            "spill specimen:",
            dict(witness.outputs),
            "events=",
            len(witness.events),
            "audio_samples=",
            witness.audio_samples,
        )


if __name__ == "__main__":
    main()
