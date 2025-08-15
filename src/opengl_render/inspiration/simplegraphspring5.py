import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import networkx as nx
import scipy as sp
import graph_express2
from graph_express2 import ProcessGraph
from graph_express2_tests import test_suite
from orbital import Orbit
import torch
import numpy as np
import random
from collections import deque
import colorsys
from binding_memranes_sympy import BindingMembrane, get_surface, mesh_from_parametric
from membrane_portal import MembranePortal
from bound_spring import BoundSpringNetwork
from particles import Visualizer
import threading, time



# target physics rate
PHYS_HZ = 240.0
PHYS_DT = 1.0 / PHYS_HZ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECOMB = 0
# Debugging mode flag: toggles fixed camera and axis visualization
DEBUG_MODE = False

# Define maximum velocity threshold
MAX_VELOCITY = 1e7  # Adjust as needed
MIN_VELOCITY = 1e2  # Adjust as needed
MAX_DT = 1
BETA_LEVEL, BETA_TYPE, BETA_ROLE, ALPHA_IDLE = 0.5, 0.7, 0.9, 0.1
LEVEL_TARGET_FACTOR, TYPE_TARGET_FACTOR, ROLE_TARGET_FACTOR = 0.9, 0.75, 0.5
# Function to refine dt based on max velocity
#this was removed unexpectedly by someone:



#this is what's left
def refine_dt(base_dt, velocities):
    refined_dt = base_dt
    while True:
        peak_velocity = torch.max(torch.norm(velocities, dim=1))
        if peak_velocity * refined_dt > MAX_VELOCITY:
            refined_dt *= 0.99  # Reduce dt iteratively
        elif refined_dt < MAX_DT and peak_velocity > 0 and peak_velocity * refined_dt < MIN_VELOCITY:
            refined_dt *= 1.01  # Increase dt
        else:
            break
    return refined_dt

SPEED_FACTOR = 1.0  # slow down simulation speed for visibility
SPRING_K, REPULSION_K, DAMPING = 0.3, 0.3, 00.870
DEFAULT_EDGE_LENGTH = 10.0

from collections import deque

def project_positions(positions):
    # no longer used: leave 3D positions for OpenGL
    return positions

def build_edges(nodes, dataG):
    edges = []
    rest_lengths = []
    for u,v in dataG.edges:
        edges.append([nodes.index(u), nodes.index(v)])
        rest_lengths.append(DEFAULT_EDGE_LENGTH)
    return torch.tensor(edges, dtype=torch.long, device=device), torch.tensor(rest_lengths, dtype=torch.float32, device=device)



import threading
import time
from double_buffer import DoubleBuffer

def rk4_step(net, positions, velocities, dt): #this isn't our algorithm but it's included for debugging convenience
    # classical RK4 on (pos, vel)
    def deriv(p, v):
        forces = net.compute_forces(p)  # you'll need to expose this
        acc = forces / net.mass - DAMPING * v
        return v, acc

    k1p, k1v = deriv(positions, velocities)
    k2p, k2v = deriv(positions + 0.5*dt*k1p, velocities + 0.5*dt*k1v)
    k3p, k3v = deriv(positions + 0.5*dt*k2p, velocities + 0.5*dt*k2v)
    k4p, k4v = deriv(positions + dt*k3p,   velocities + dt*k3v)

    positions = positions + dt*(k1p + 2*k2p + 2*k3p + k4p)/6
    velocities = velocities + dt*(k1v + 2*k2v + 2*k3v + k4v)/6
    return positions, velocities
class PhysicsEngine:
    def __init__(self, net):
        self.net = net
        self.engine = BoundSpringHelper(self.net)

    def step(self, dt):
        
        # For debugging convenience, use RK4 step
        #new_positions, new_velocities = rk4_step(self.net, positions, velocities, dt)
        #self.net.set_state(new_positions, new_velocities)

        # Alternatively, use the network's step method if available
        positions, velocities = self.engine.step(dt)
        return positions, velocities

class BoundSpringHelper:
    def __init__(self, net):
        self.net = net
        self.membrane = None
        self.membrane_verts = None
        self.membrane_faces = None
        self.nodes = []
        self.positions = None
        self.velocities = None
        self.edges = None
        self.base_lengths = None
        self.N = 0
        self.E = 0
        self.typ_mask = None
        self.role_mask = None
        self.lvl_mask = None
        self.node_lvl = None
        self.node_typ = None
        self.node_role = None
        self.net = ProcessGraphHelper(self.net)
        self.initialize_bound_spring_network(self.net)
        self.frame = 0
        

    def initialize_bound_spring_network(self, net):
        self.membrane = BindingMembrane(centre=(0,0,0), radius=50.0)
        sph = get_surface("unit_sphere")
        verts_np, faces_np = mesh_from_parametric(sph, 32)
        self.membrane_verts = torch.tensor(verts_np, dtype=torch.float32, device=device) * self.membrane.radius
        self.membrane_faces = torch.tensor(faces_np, dtype=torch.long, device=device)
        portal = MembranePortal(self.membrane, self.membrane_verts, self.membrane_faces)

        def build_graph(pg):
            net = BoundSpringNetwork(
                pg,
                k_stretch=SPRING_K,
                c_repulse=REPULSION_K,
                damping=DAMPING
            )

            # === now defer entirely to net for all vertex data ===
            nodes      = net.node_labels()     # e.g. ['A','B',...]
            positions  = net.pos.clone().to(device) # tensor shape [N,3]
            velocities = net.vel.clone().to(device)
            

            edges           = net.edge_list()  # list of (u,v) tuples
            base_lengths = net.rest_lengths.clone().to(device)
            N, E            = positions.shape[0], len(edges)
            lvl_mask, node_lvl   = net.level_mask()
            typ_mask, node_typ   = net.type_mask()
            role_mask, node_role = net.rol_mask()
            # Move masks to device
            lvl_mask = lvl_mask.to(device)
            typ_mask = typ_mask.to(device)
            role_mask = role_mask.to(device)
            node_lvl = node_lvl.to(device)
            node_typ = node_typ.to(device)
            node_role = node_role.to(device)
            base_lengths = net.rest_lengths.clone().to(device)
            return net, nodes, positions, velocities, edges, base_lengths, N, E,\
                    typ_mask, role_mask, lvl_mask, node_lvl, node_typ, node_role,
        self.net, self.nodes, self.positions, self.velocities, self.edges, self.base_lengths, self.N, self.E,\
        self.typ_mask, self.role_mask, self.lvl_mask, self.node_lvl, self.node_typ, self.node_role = build_graph(net)

        self.build_graph = build_graph

        self.build_set_masks(self)
        self.articulated_base_lengths = self.base_lengths.clone().to(device)

    def build_set_masks(self):
        """Build masks for node and edge types, roles, and levels.
        """
        
        # Precompute role masks
        self.edge_keys = [(self.nodes[u],self.nodes[v]) for u,v in self.edges]

        self.edge_rest_lengths = self.net.rest_lengths.clone().to(device)  # edge rest lengths
        for idx,(lvl,typ,role) in enumerate(self.process_graph.ordered_keys):
            self.lvl_set, self.typ_set, self.role_set = self.process_graph.edges_at_level(lvl), self.process_graph.edges_at_type(lvl,typ), set(self.process_graph.grouped[lvl][typ].get(role,[]))
            for e_idx,key in enumerate(self.edge_keys):
                if key in self.lvl_set: self.lvl_mask[idx][e_idx]=1
                if key in self.typ_set: self.typ_mask[idx][e_idx]=1
                if key in self.role_set: self.role_mask[idx][e_idx]=1
            for n_idx,nid in enumerate(self.nodes):
                if any(nid==u or nid==v for (u,v) in self.lvl_set): self.node_lvl[idx][n_idx]=1
                if any(nid==u or nid==v for (u,v) in self.typ_set): self.node_typ[idx][n_idx]=1
                if any(nid==u or nid==v for (u,v) in self.role_set): self.node_role[idx][n_idx]=1

    def apply_auto_freeze(self, ordered_keys, edges_at_role):
        
        # Get the first level, type, and role group
        first_level, first_type, first_role = ordered_keys[0]
        first_edges = edges_at_role(first_role)
        first_sources = {u for u, v in first_edges}
        first_destinations = {v for u, v in first_edges}

        # Get the last level, type, and role group
        last_level, last_type, last_role = ordered_keys[-1]
        last_edges = edges_at_role(last_role)
        last_destinations = {v for u, v in last_edges}

        # Update the fixed mask
        self.fixed_mask.zero_()
        for node in first_sources.union(first_destinations).union(last_destinations):
            self.fixed_mask[self.nodes.index(node)] = True

        return self.fixed_mask
    
    @torch.no_grad()
    def step(self, dt: float = 1/60.):
        self.positions, self.velocities = self.net.step(dt)
        # Apply auto-freeze if enabled
        if hasattr(self, 'auto_freeze_enabled') and self.auto_freeze_enabled:
            self.fixed_mask = self.apply_auto_freeze(self.ordered_keys, self.edges_at_role)
        return self.positions, self.velocities


class GraphObject:
    def __init__(self, net):
        self.net = self.iniatialize_network(net)

    def initialize_network(self):
        ...

class ProcessGraphHelper(ProcessGraph, GraphObject):

    def __init__(self, net=None, expr=None, recomb=None, demo=None, *dims):

        """ Initialize a ProcessGraph with optional network, expression, and recombination.
        If no network is provided, a new one is created with the given recombination factor.
        If no expression is provided, it defaults to the first demo in the test suite.
        """
        
        if not expr:
            demo = test_suite[demo | 0] if isinstance(demo, int) else {'expr_fn': sp.Expr(sp.Symbol('x')+sp.Symbol('y')*sp.Symbol('z')), 'dims': (10, 10, 10)}
            expr = demo.get('expr_fn') or demo
        recomb = recomb | RECOMB
        dims = demo.get('dims', None)
        if not recomb:
            recomb = demo.get('recomb', 0)
        if net:
            self.net = net
        else:
            self.net = super().__init__(recomb)
        GraphObject.__init__(self, net)
        self.expr = expr
        self.dims = dims

        self.initialize_network()

        self.first_level = self.ordered_keys[0][0]   # 0
        self.last_level  = self.ordered_keys[-1][0]  # e.g. 6

        self.initial_inputs = self.edges_at_level_role(self.first_level, "input")
        self.last_outputs   = self.edges_at_level_role(self.last_level,  "output")

        self.default_subsets = set.union(self.initial_inputs, self.last_outputs)


    def initialize_network(self):
        """
        Build the process graph from an expression.
        """
    
        super().build_from_expression(self.expr, *self.dims)
        self.compute_levels(method='alap')
        if not self.dataG:
            print("Graph is empty")
            return None, None
        self.grouped = self.group_edges_by_dataset(self.dataG)
        self.ordered_keys = self.sort_roles(self.grouped)
        return self.net

    def edges_at_level(self, l): return set.union(*(set(self.grouped[l][t][r]) for t in self.grouped[l] for r in self.grouped[l][t]))
    def edges_at_type(self, l, t): return set.union(*(set(self.grouped[l][t][r]) for r in self.grouped[l][t]))
    def edges_at_role(self, r): return set.union(*(set(self.grouped[l][t].get(r, [])) for l in self.grouped for t in self.grouped[l]))
    def edges_at_level_role(self, level, role):
        """Union of all edges at `level` (across every type) having `role`."""
        if level not in self.grouped:           # defensive-guard: level might be empty
            return set()
        # collect every list matching the role, defaulting to [] when absent
        iterables = (self.grouped[level][t].get(role, []) for t in self.grouped[level])
        return set().union(*iterables) if self.grouped[level] else set()

class GraphPhysVisUpdater:
    """
    Base class for updating graph physics and visualization.
    """
    def __init__(self, double_buffer):
        self.double_buffer = double_buffer

    def full_reset(self, menu_index, phys_object, vis_object):
        if not phys_object or not vis_object:
            raise ValueError("Physics and visualization objects must be provided.")
        while True:
            try:
                demo = test_suite[menu_index]
                expr = demo.get('expr_fn') or demo
                recomb = demo.get('recomb', RECOMB)


                phys_object.net = ProcessGraphHelper(expr=expr, demo=demo, recomb=recomb)
                phys_object.engine = BoundSpringHelper(phys_object.net)
                
                vis_object.reset_buffers(phys_object)

                break  # success

            except Exception as e:
                print(f"Initialization failed: {e}. Skipping to next test.")
                menu_index = (menu_index + 1) % len(test_suite)
                continue  # try next demo


    def physics_worker(self, double_buffer, stop_event, net, *args, positions=None, velocities=None):
        """
        Physics simulation thread.
        """
        phys_engine = PhysicsEngine(net)

        phys_engine.set_dt(PHYS_DT)  # set the target dt for the network
        while not stop_event.is_set():
        
            dt = phys_engine.dt
            sleep_correction = time.time()
            new_positions, new_velocities = phys_engine.step(dt)
            sleep_correction = time.time() - sleep_correction
            # Write updated positions to double buffer
            double_buffer.write(new_positions.cpu().numpy())
            double_buffer.swap()
            # Optionally sleep to control update rate
            if (dt- sleep_correction) > 0:
                time.sleep(dt- sleep_correction)  # adjust sleep to maintain target dt
            

    def render_worker(self, double_buffer, stop_event, vis_object = None, *args):
        """
        Rendering thread.
        """
        if not vis_object:
            vis_object = visualizer
        while not stop_event.is_set():
            # Read latest positions from double buffer
            positions = double_buffer.read()
            vis_object.step(positions)
            time.sleep(1.0 / vis_object.FPS)

    def deploy_workers(self):
        """
        Deploy physics and rendering threads.
        """
        if not hasattr(self, 'double_buffer'):
            raise ValueError("Double buffer must be initialized before deploying workers.")

        # Create stop event for thread termination
        stop_event = threading.Event()

        # Start physics worker thread
        phys_thread = threading.Thread(target=self.physics_worker, args=(self.double_buffer, stop_event, self.net))
        phys_thread.start()

        # Start rendering worker thread
        render_thread = threading.Thread(target=self.render_worker, args=(self.double_buffer, stop_event, visualizer))
        render_thread.start()

        return phys_thread, render_thread, stop_event

double_buffer = DoubleBuffer()  # Initialize with dummy data
reset_manager = GraphPhysVisUpdater(double_buffer)
visualizer = Visualizer(FPS=60, window_size=(800, 600), title="Graph Spring Simulation", reset_manager=reset_manager)
phys_obj = PhysicsEngine()
reset_manager.full_reset(0, phys_obj, visualizer)

def main():
    reset_manager.deploy_workers()


if __name__=="__main__":
    main()

