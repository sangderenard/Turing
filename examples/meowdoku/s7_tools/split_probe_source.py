def damped_axis(x: float, v: float, axis: float, target: float, lo: float, hi: float,
                speed: float, tau: float, reach: float, dt: float):
    steering = (target >= 0.0) * 1.0
    gap = (target - x) / reach
    pointer = (abs(gap + speed) - abs(gap - speed)) * 0.5
    keys = (abs(axis + 1.0) - abs(axis - 1.0)) * 0.5 * speed
    command = steering * pointer + (1.0 - steering) * keys
    moved = v + (command - v) * (dt / (tau + dt))
    free = x + moved * dt
    x_next = (abs(free - lo) - abs(free - hi) + lo + hi) * 0.5
    v_next = (x_next - x) / dt
    return x_next, v_next


def ship_frame(x: float, y: float, vx: float, vy: float, ax: float, ay: float,
               frame: float, substeps: int):
    dt = frame / substeps
    for _ in range(substeps):
        x, vx = damped_axis(x, vx, ax, -1.0, 0.3, 12.2, 6.0, 0.08, 0.25, dt)
        y, vy = damped_axis(y, vy, ay, -1.0, 0.7, 4.0, 6.0, 0.08, 0.25, dt)
    return x, y, vx, vy
