from math import pi

def sphere_radius_from_volume(V: float) -> float:
    return (3.0*max(V, 0.0)/(4.0*pi))**(1.0/3.0)

def sphere_area_from_volume(V: float):
    R = sphere_radius_from_volume(V)
    return 4.0*pi*R*R, R
