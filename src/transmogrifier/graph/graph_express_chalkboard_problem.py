import sympy as sp


# --- 1) Symbols & “arrays” ---
# sizes
N, E, n_sub = sp.symbols('N E n_sub', integer=True, positive=True)
# time, physics params
dt, growth_rate, damping = sp.symbols('dt growth_rate damping', positive=True)
k_stretch, c_repulse = sp.symbols('k_stretch c_repulse', real=True)
max_force, max_velocity, max_disp = sp.symbols('max_force max_velocity max_disp', positive=True)
tol = sp.symbols('tol', positive=True)

# c_frac and c_transients
c_frac, k_transient = sp.symbols('c_frac k_transient', positive=True)

# arrays / tensors
pos     = sp.IndexedBase('pos')       # pos[v,i]
vel     = sp.IndexedBase('vel')       # vel[v,i]
rest    = sp.IndexedBase('rest')      # rest[e]
base    = sp.IndexedBase('base')      # base[e]
mass    = sp.IndexedBase('mass')      # mass[v]
normals = sp.IndexedBase('normals')   # normals[v,i]
hullv   = sp.IndexedBase('hullv')     # hull vertices, hullv[h,i]

# edge indices
u, v = sp.symbols('u v', integer=True)
edges = sp.IndexedBase('edges')       # edges[e,0]=u, edges[e,1]=v

# loop indices
e, i, d, j, s = sp.symbols('e i d j s', integer=True)

# --- 2) spring + repulsion force for each vertex i ---
# Spring: sum over edges touching i
# define a helper that for each edge e contributes to both endpoints
spring_contrib = (
    k_stretch
    * (( sp.sqrt(sp.summation((pos[edges[e,0],d] - pos[edges[e,1],d])**2, (d,0,2))) 
         - rest[e]
       )
      / (sp.sqrt(sp.summation((pos[edges[e,0],d] - pos[edges[e,1],d])**2, (d,0,2))) + 1e-9)
    )
    * (pos[edges[e,0],i] - pos[edges[e,1],i])
)
F_spring_i = sp.Sum(spring_contrib, (e,0,E-1))

# Drag/repulsion: simple form c_repulse * vel
F_drag_i = -c_repulse * vel[i,0]  # you'd vectorize this similarly

# --- 3) boundary piecewise force for each vertex i (radial slip) ---
# distance to hull-center (assumed zero for simplicity)
r_i   = sp.sqrt(sp.summation(pos[i,d]**2, (d,0,2)))
r_h   = sp.symbols('r_h', positive=True)
# Piecewise: if inside inner_transient region, spring reject; else 0
F_bound_i = sp.Piecewise(
    ( -k_transient*(sp.symbols('inner_transient') - r_i)*(pos[i,0]/(r_i+1e-9)), r_i < sp.symbols('inner_transient') ),
    ( 0, True )
)

# --- 4) total force & relativistic accel ---
F_i = F_spring_i + F_drag_i + F_bound_i
a_i = F_i / mass[i]
gamma_i = 1/sp.sqrt(1 - (sp.sqrt(sp.summation(vel[i,d]**2,(d,0,2))) / (c_frac*max_velocity))**2)
a_rel_i = a_i / gamma_i**3

# --- 5) per-vertex dt negotiation & n_steps_i ---
dt_f_i = max_force    / (sp.Abs(F_i) + 1e-9)
dt_a_i = max_velocity / (sp.Abs(a_rel_i) + 1e-9)
dt_v_i = max_disp     / (sp.Abs(sp.sqrt(sp.summation(vel[i,d]**2,(d,0,2)))) + 1e-9)
dt_allowed_i = sp.Min(dt_f_i, dt_a_i, dt_v_i)
n_steps_i    = sp.ceiling(dt / dt_allowed_i)

# --- 6) global n_sub as mean over i=0..N-1 ---
n_sub_expr = sp.ceiling( (1/N) * sp.summation(n_steps_i, (i,0,N-1)) )

# --- 7) integrate over sub‐steps s=1..n_sub_expr ---
# pos_next[i,d] = pos[i,d] + Sum_{s=1..n_sub} (exp(-damping*dt_sub)*vel[i,d] + a_rel_i*dt_sub)
dt_sub = dt / n_sub_expr
vel_term = sp.exp(-damping*dt_sub)*vel[i,d]
pos_next = pos[i,d] + sp.summation((vel_term + a_rel_i*dt_sub), (s,1,n_sub_expr))

# --- 8) bundle into one master expression ---
master_expr = pos_next
chalkboard_problem = sp.Eq(pos[i,d], master_expr)

if __name__ == "__main__":
    # Print the chalkboard problem for verification
    print("Chalkboard Problem Expression:")
    sp.pprint(chalkboard_problem)
    print("\nSymbolic Variables:")
    sp.pprint(sp.lambdify((pos, vel, rest, base, mass, normals, hullv, edges, n_sub), chalkboard_problem))
    from .graph_express2 import ProcessGraph
    from .graph_deep_compiler import GraphDeepCompiler
    from operator_defs import torch_funcs, torch_sigs
    pg = ProcessGraph()
    pg.build_from_expression(chalkboard_problem)
    pg.compute_levels("asap")
    compiler = GraphDeepCompiler(pg, torch_funcs, torch_sigs)
    f = compiler.build_function()
    compiler.print_source()
    
