import numpy as np
import torch
import colorsys
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_f
from collections import deque
import ctypes
from graph_express2_tests import test_suite
# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLU import *

# Visualization and rendering constants (moved from simplegraphspring5.py)
CANVAS_COLOR = (0.1, 0.1, 0.1, 1.0)  # Background color
EDGE_ALPHA = 0.3  # default edge alpha blending

BUFFER_SIZE = 300  # number of recent frames to keep
ACTIVE_EDGE_COUNT = 5  # number of top active edges to consider per frame
GHOST_COUNT = 5  # number of ghosts to keep in rainbow trail
GHOST_INIT_ALPHA = 0.5
GHOST_MIN_DECAY = 0.90
GHOST_MAX_DECAY = 0.99
GHOST_PEAK_RADIUS = 0.3  # radius for ghost nodes

ENABLE_RAINBOW_TRAIL = True  # toggle rainbow ghost trail
TRAIL_COUNT = GHOST_COUNT
TRAIL_PRUNE_THRESHOLD = 0.01  # minimum alpha before pruning
TRAIL_MODE = "trail"
BEST_MODE  = "best"
RAINBOW_MODE = TRAIL_MODE  # set to TRAIL_MODE or BEST_MODE

BEST_SCALES = [10, 50, 250, 1000, 5000]  # or as desired, powers-of-n
BEST_TIMEOUT = 50   # Don't admit bests before this many frames

# --- order-strip ----------------------------------------------------------
ORDER_SMOOTH_ALPHA = 0.15   # EMA smoothing
STRIP_Y_NDC        = -0.92  # vertical NDC position (−1 = bottom)
STRIP_POINT_SIZE   = 6.0    # base GL_POINT size
STRIP_Y_NDC     = -0.92    # vertical bar position
STRIP_PIX_SCALE = 20.0     # exactly the same “radius→pixels” you use in 3-D


WIDTH, HEIGHT = 800, 600
NODE_RADIUS = .15
FPS = 60
EDGE_BASE_WIDTH, EDGE_GLOW_WIDTH = 2, 2
GLOW_RISE, GLOW_DECAY = 0.3, 0.05
GLOW_PEAK_ALPHA, GLOW_FLOOR_ALPHA = 1.0, 0.1
GLOW_PEAK_RADIUS, GLOW_FLOOR_RADIUS = 0.2, .10
CAMERA_DISTANCE, FOCAL_LENGTH = 100, 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STRIP_VERT = """
#version 330
in  float in_order;      // 0‥1 rank along the bar
in  vec3  in_color;
in  float in_alpha;
in  float in_radius;
in  float in_ke;

out vec3  v_color;
out float v_alpha;

uniform float u_y;       // NDC Y position
uniform float u_px;      // base px size per radius unit

void main() {
    // 2-D bar coordinate (x from order, y fixed)
    gl_Position = vec4(2.0*in_order - 1.0, u_y, 0.0, 1.0);
    gl_PointSize = in_radius * u_px + in_ke * 5.0;
    v_color  = in_color;
    v_alpha  = in_alpha;
}
"""

STRIP_FRAG = """
#version 330
in  vec3  v_color;
in  float v_alpha;
out vec4  out_color;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (length(d) > 0.5) discard;        // round dot
    float a = v_alpha * smoothstep(0.5, 0.45, length(d));
    out_color = vec4(v_color, a);
}
"""


VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
in float in_alpha;
in float in_radius;
in float in_ke;

out vec3 frag_color;
out float frag_alpha;
out float frag_ke;
uniform mat4 u_mvp;
uniform float u_time;

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1,0,0,  0,c,-s,  0,s,c);
}
mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s,  0,1,0,  -s,0,c);
}
mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,-s,0, s,c,0, 0,0,1);
}


void main() {
    // apply combined MVP passed from CPU
    float t = u_time * 0.0001; 
    vec3 rotated = rotX(t) * rotY(0.8*t) * rotZ(1.5*t) * in_position;
    gl_Position = u_mvp * vec4(rotated, 1.0);

    gl_PointSize = in_radius * 20.0 + in_ke * 5.0; // customizable
    frag_color = in_color;
    frag_alpha = in_alpha;
    frag_ke = in_ke;
}
"""
DEFAULT_INIT_RADIUS = 10.0  # default radius for initial node positions

FRAGMENT_SHADER = """
#version 330
in vec3 frag_color;
in float frag_alpha;
in float frag_ke;

out vec4 out_color;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    // smooth edge for anti-aliasing
    float alpha = frag_alpha * smoothstep(0.5, 0.45, dist);
    out_color = vec4(frag_color, alpha);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_program(vertex_src, fragment_src):
    program = glCreateProgram()
    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    return program

def setup_vbo():
    vbo = glGenBuffers(1)
    return vbo

def update_vbo(vbo, data):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

def pca1_scores(X: torch.Tensor) -> torch.Tensor:
    """Return centered projection onto the first principal axis (N,)."""
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    return (Xc @ pc1)

class Visualizer:
    def __init__(self, FPS=FPS, window_size=(800, 600), reset_manager=None, N=0, E=0, title="Visualizer"):
        self.FPS = FPS
        self.N = N
        self.E = E
        self.window_size = window_size
        self.title = title
        self.clock = pygame.time.Clock()
        self.frame = 0
        self.reset_manager = reset_manager

        # BUFFER_SIZE is already defined above
        self.state_buffer   = deque(maxlen=BUFFER_SIZE)
        self.best_score     = float('inf')
        self.ghost_configs  = []
        self.auto_freeze_enabled = False  # Toggle for auto-freezing nodes
        self.fixed_mask = torch.zeros(0, dtype=torch.bool, device=device)  # Will be set in full_reset
        # TRAIL_LENGTH must match what you use inside main()
        self.TRAIL_LENGTH   = 24
        self.rainbow_trail  = deque(maxlen=GHOST_COUNT)

        # VBO handles (initialized later via setup_vbo)
        self.vbo_nodes        = None
        self.vbo_node_colors  = None
        self.vbo_edges        = None
        self.vbo_edge_colors  = None
        self.vbo_ghosts       = None

        self.decay_speeds = np.linspace(GHOST_MAX_DECAY, GHOST_MIN_DECAY, GHOST_COUNT)
        self.rainbow_colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0)
                        for h in np.linspace(0, 1, GHOST_COUNT, endpoint=False)]
        # For trail mode: a fixed-length deque of past frame positions
        self.TRAIL_LENGTH = 24
        self.rainbow_trail = deque(maxlen=self.TRAIL_LENGTH)
        self.trail_colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0, 1, self.TRAIL_LENGTH, endpoint=False)]

        # For best mode: as before, but persistent, not fading
        self.BEST_SCALES = [10, 50, 250, 1000, 5000]
        self.best_per_scale = [None] * len(self.BEST_SCALES)
        self.hues = np.linspace(0, 1, len(self.BEST_SCALES), endpoint=False)
        self.rainbow_colors = [colorsys.hsv_to_rgb(h, 1, 1) for h in self.hues]

        # Select demo from graph_express2.test_suite

        
        self.best_per_scale = [None] * len(BEST_SCALES)  # stores dicts: {'score', 'frame', 'positions'}
        self.best_per_scale_trail = [None] * len(BEST_SCALES)  # for rainbow ghosts

        # Setup OpenGL
        pygame.init()
        pygame.font.init()
        title_font = pygame.font.SysFont('Arial', 20)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 8)
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF|OPENGL)
        glEnable(GL_MULTISAMPLE)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, WIDTH/HEIGHT, 1.0, 1000.0)
        gluLookAt(0, 0, 100.0,   0, 0, 0,   0, 1, 0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 100.0,   0, 0, 0,   0, 1, 0)
        glEnable(GL_DEPTH_TEST)
        # allow shader-controlled point sizes for larger vertices
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_POINT_SMOOTH)
        # enable blending for proper round point rendering
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        vbo_nodes = setup_vbo()
        vbo_node_colors = setup_vbo()
        vbo_edges = setup_vbo()
        vbo_edge_colors = setup_vbo()
        vbo_ghosts = setup_vbo()

        shader_program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
        strip_prog   = create_program(STRIP_VERT, STRIP_FRAG)
        order_scores = torch.zeros(N, device=device)   # after N is known

        strip_vbo   = setup_vbo()

        loc_ord  = glGetAttribLocation(strip_prog, "in_order")
        loc_col  = glGetAttribLocation(strip_prog, "in_color")
        loc_alp  = glGetAttribLocation(strip_prog, "in_alpha")
        loc_rad  = glGetAttribLocation(strip_prog, "in_radius")
        loc_ke   = glGetAttribLocation(strip_prog, "in_ke")

        loc_y    = glGetUniformLocation(strip_prog, "u_y")
        loc_px   = glGetUniformLocation(strip_prog, "u_px")

        # retrieve MVP and time uniform locations
        mvp_loc = glGetUniformLocation(shader_program, "u_mvp")
        time_loc = glGetUniformLocation(shader_program, "u_time")
        clock = pygame.time.Clock()
        frame = 0
        # vertex count for drawing edges (two vertices per edge)
        edge_count = E * 2
        menu_index = 0  # Track the current menu index




    def reset_buffers(self, N, E):
        self.order_scores = torch.zeros(N, device=device)   # after N is known
        self.glow_alpha  = torch.full((N,1), GLOW_FLOOR_ALPHA, dtype=torch.float32, device=device)
        self.glow_radius = torch.full((N,1), GLOW_FLOOR_RADIUS, dtype=torch.float32, device=device)
        # 2) Clear rolling/state buffers
        self.state_buffer.clear()
        #state_buffer.append({'sum': 0, 'edges': [], 'positions': positions.clone()})
        self.best_score = float('inf')
        self.ghost_configs.clear()
        self.rainbow_trail.clear()

        # 3) Re-allocate (or clear) all VBOs to match new sizes
        import numpy as np

        # Nodes: 9 floats per vertex
        empty_node_data = np.zeros((N, 9), dtype=np.float32)
        update_vbo(self.vbo_nodes, empty_node_data)

        # (If used) separate color buffer for nodes
        empty_node_colors = np.zeros((N, 3), dtype=np.float32)
        update_vbo(self.vbo_node_colors, empty_node_colors)

        # Edges: 2 endpoints × 2 floats each, plus RGBA colors
        empty_edge_pts   = np.zeros((E * 2, 2), dtype=np.float32)
        empty_edge_cols  = np.zeros((E * 2, 4), dtype=np.float32)
        update_vbo(self.vbo_edges, empty_edge_pts)
        update_vbo(self.vbo_edge_colors, empty_edge_cols)

        # Ghosts: same layout as nodes
        update_vbo(self.vbo_ghosts, np.zeros((0, 9), dtype=np.float32))
    def step(self, buffer_page, buffer_map):
        state_dict = read_buffer(buffer_page, buffer_map)


        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit()
                return
            elif e.type == KEYDOWN:
                if e.key == K_LEFT:
                    menu_index = (menu_index - 1) % len(test_suite)
                    self.reset_manager.full_reset(menu_index)
                elif e.key == K_RIGHT:
                    menu_index = (menu_index + 1) % len(test_suite)
                    self.reset_manager.full_reset(menu_index)
                elif e.key == K_f:  # Toggle freeze
                    auto_freeze_enabled = not auto_freeze_enabled
                    if auto_freeze_enabled:
                        self.apply_auto_freeze()
                    else:
                        fixed_mask.zero_()
                current_active_sum = active_lengths.sum().item()
        # Buffer detailed state: sum, active edge indices, and positions snapshot
        self.state_buffer.append({
            'sum': current_active_sum,
            'edges': active_edges.cpu().tolist(),
            'positions': positions.clone()
        })
        # Rolling sum over the buffered sums
        rolling_sum = sum(item['sum'] for item in self.state_buffer)
        # when rolling sum hits new low, record ghost state
        if rolling_sum < self.best_score:
            self.best_score = rolling_sum
            # select next ghost index cyclically
            idx_g = len(self.ghost_configs) % GHOST_COUNT
            self.ghost_configs.append({
                'positions': positions.clone(),
                'alpha': GHOST_INIT_ALPHA,
                'decay': float(self.decay_speeds[idx_g]),
                'color': self.rainbow_colors[idx_g]
            })
            if len(self.ghost_configs) > GHOST_COUNT:
                self.ghost_configs.pop(0)
            print(f"New best rolling sum: {self.best_score} at frame {frame}")

        # instead of local glow_alpha/radius, strip that logic out of main
        # and simply draw from net:

        # fetch all visuals from net:
        colors       = net.node_colors()     # numpy Nx3
        alpha_np     = net.glow_alpha.cpu().numpy()    # Nx1
        radius_np    = net.glow_radius.cpu().numpy()   # Nx1
        ke_np        = net.kinetic_energy().reshape(-1,1).cpu().numpy()  # Nx1

        node_data = np.concatenate(
            [ positions.cpu().numpy(),  # Nx3
            colors,                  # Nx3
            alpha_np,                # Nx1
            radius_np,               # Nx1
            ke_np ], axis=1)         # → Nx9
        update_vbo(vbo_nodes, node_data)

        # … inside your main simulation loop, *after* velocities are
        # updated but *before* you commit `positions += velocities*dt`:



        # Restore fixed positions if frozen
        if auto_freeze_enabled:
            apply_auto_freeze()
        # --- Derived values ---
        # project to pixel-space for edges and normalize to NDC for shaders
        
        PCA = True
        if PCA:
            # ---- PCA-1 ordering ------------------------------------------------------
            scores_raw = pca1_scores(positions)                   # (N,)
            if order_scores is None or order_scores.numel() != N:
                order_scores = scores_raw.clone().to(device)
            else:
                order_scores = (1-ORDER_SMOOTH_ALPHA)*order_scores + \
                            ORDER_SMOOTH_ALPHA*scores_raw.to(device)

            rank          = order_scores.argsort()                # low→high
            order_coord   = torch.zeros_like(order_scores, dtype=torch.float32)
            order_coord[rank] = torch.linspace(0, 1, N).to(device)           # 0-1 monotone
            order_np      = order_coord.unsqueeze(1).cpu().numpy().astype(np.float32)
            update_vbo(strip_vbo, order_np)        # <-- one float per node

        # glow updates
        alpha_target = GLOW_FLOOR_ALPHA + (GLOW_PEAK_ALPHA-GLOW_FLOOR_ALPHA) * \
                       (0.3*node_lvl[idx] + 0.5*node_typ[idx] + 1.0*node_role[idx]).unsqueeze(1)
        radius_target = GLOW_FLOOR_RADIUS + (GLOW_PEAK_RADIUS-GLOW_FLOOR_RADIUS) * \
                       (0.3*node_lvl[idx] + 0.5*node_typ[idx] + 1.0*node_role[idx]).unsqueeze(1)
        glow_alpha = glow_alpha.to(device)
        glow_radius = glow_radius.to(device)
        alpha_target = alpha_target.to(device)
        radius_target = radius_target.to(device)
        glow_alpha += (alpha_target - glow_alpha) * torch.where(alpha_target > glow_alpha, GLOW_RISE, GLOW_DECAY)
        glow_radius += (radius_target - glow_radius) * torch.where(radius_target > glow_radius, GLOW_RISE, GLOW_DECAY)

        # --- Prepare interleaved node data with proper colors and 3D positions ---
        # world-space positions
        pos_np = positions.cpu().numpy().astype(np.float32)  # N x 3
        # compute node colors for this frame
        colors      = net.node_colors()              # shape [N,3]
        alpha_np    = net.glow_alpha.reshape(-1,1).cpu().numpy()  # N x 1
        radius_np   = net.glow_radius.reshape(-1,1).cpu().numpy()  # N x 1
        ke_np       = net.kinetic_energy().reshape(-1,1).cpu().numpy()  # N x 1
        # concatenate into N x 9 array: pos(3), color(3), alpha, radius, ke
        
        node_data = np.concatenate([pos_np, colors, alpha_np, radius_np, ke_np], axis=1)
        update_vbo(vbo_nodes, node_data)
        if PCA:
            # ---------- strip data (order + live state) ------------------------------
            strip_data = np.hstack([                         # N × 9  (1+3+1+1+1+2 pad)
                order_coord.unsqueeze(1).cpu().numpy(),      # order
                colors,                                      # same RGB already computed
                alpha_np,                                    # glow_alpha
                radius_np,                                   # glow_radius
                ke_np                                        # kinetic energy
            ]).astype(np.float32)
            update_vbo(strip_vbo, strip_data)

        # --- Legacy edge drawing data ---
        # draw edges in pixel-space to align with orthographic projection
        edge_points = np.vstack([proj_pixels[edges[:,0]].cpu().numpy(), proj_pixels[edges[:,1]].cpu().numpy()]).astype(np.float32)
        # build edge color RGBA for blending
        edge_rgb = np.vstack([torch.stack([role_mask[idx], typ_mask[idx], lvl_mask[idx]], axis=1).cpu().numpy()] * 2).astype(np.float32)
        edge_alphas = np.full((edge_rgb.shape[0], 1), EDGE_ALPHA, dtype=np.float32)
        edge_colors = np.hstack([edge_rgb, edge_alphas])
        update_vbo(vbo_edges, edge_points)
        update_vbo(vbo_edge_colors, edge_colors)

        # --- Render ---
        glClearColor(*CANVAS_COLOR)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set camera for shader & debugging
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if DEBUG_MODE:
            # fixed eye positioned along Z to see entire graph
            #gluLookAt(0, 0, 300.0,   0, 0, 0,   0, 1, 0)
            # draw world axes
            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(100, 0, 0)
            # Y axis (green)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 100, 0)
            # Z axis (blue)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 100)
            glEnd()

        # ---- Draw edges with legacy pipeline ----
        glEnableClientState(GL_VERTEX_ARRAY)
        glLineWidth(EDGE_BASE_WIDTH * 2)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_edges)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_edge_colors)
        glColorPointer(4, GL_FLOAT, 0, None)  # RGBA
        #glDrawArrays(GL_LINES, 0, edge_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # ---- Draw nodes with shader pipeline ----
        # compute and upload combined ModelViewProjection matrix
        proj_mat = (GLfloat * 16)()
        model_mat = (GLfloat * 16)()
        glGetFloatv(GL_PROJECTION_MATRIX, proj_mat)
        glGetFloatv(GL_MODELVIEW_MATRIX, model_mat)
        # convert to numpy, reshape and transpose to column-major
        proj_np = np.array(proj_mat, dtype=np.float32).reshape(4,4)
        model_np = np.array(model_mat, dtype=np.float32).reshape(4,4)
        mvp_np = proj_np.dot(model_np)
        # upload uniform
        glUseProgram(shader_program)
        # send time uniform for rotation
        current_time = pygame.time.get_ticks()
        glUniform1f(time_loc, current_time)
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_np.flatten())

        glBindBuffer(GL_ARRAY_BUFFER, vbo_nodes)

        stride = 9 * 4  # 9 floats (pos3 + color3 + alpha + radius + ke) * 4 bytes
        # in_position (vec3)
        loc = glGetAttribLocation(shader_program, "in_position")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # in_color (vec3 based on role/type/level masks)
        loc_c = glGetAttribLocation(shader_program, "in_color")
        glEnableVertexAttribArray(loc_c)
        glVertexAttribPointer(loc_c, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3*4))
        # in_alpha
        loc_a = glGetAttribLocation(shader_program, "in_alpha")
        glEnableVertexAttribArray(loc_a)
        glVertexAttribPointer(loc_a, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6*4))
        # in_radius
        loc_r = glGetAttribLocation(shader_program, "in_radius")
        glEnableVertexAttribArray(loc_r)
        glVertexAttribPointer(loc_r, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(7*4))
        # in_ke
        loc_k = glGetAttribLocation(shader_program, "in_ke")
        glEnableVertexAttribArray(loc_k)
        glVertexAttribPointer(loc_k, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8*4))

        glDrawArrays(GL_POINTS, 0, N)

        glDisableVertexAttribArray(loc)
        glDisableVertexAttribArray(loc_c)
        glDisableVertexAttribArray(loc_a)
        glDisableVertexAttribArray(loc_r)
        glDisableVertexAttribArray(loc_k)
        glUseProgram(0)


        if not ENABLE_RAINBOW_TRAIL:
            pass
        elif RAINBOW_MODE == TRAIL_MODE:
            rainbow_trail.append({
                'positions': positions.clone(),
                'frame': frame  # or pygame.time.get_ticks() for ms-precise
            })

        elif RAINBOW_MODE == BEST_MODE:
            
            # Update best per scale, but do not fade, always persistent
            for i, window in enumerate(BEST_SCALES):
                if len(state_buffer) >= window:
                    window_scores = [item['sum'] for item in list(state_buffer)[-window:]]
                    window_min = min(window_scores)
                    min_idx = window_scores.index(window_min)
                    min_item = list(state_buffer)[-window + min_idx]
                    # Only update if this is a new best for the window
                    last_best = best_per_scale[i]
                    if last_best is None or window_min < last_best['score']:
                        best_per_scale[i] = {
                            'score': window_min,
                            'frame': frame,
                            'positions': min_item['positions'].clone()
                        }

        # After updating trail or best buffers:
        ghost_pos, ghost_col, ghost_alpha, ghost_age = [], [], [], []
        if not ENABLE_RAINBOW_TRAIL:
            pass
        elif RAINBOW_MODE == TRAIL_MODE:
            for i, trail_state in enumerate(rainbow_trail):
                pos = trail_state['positions'].cpu().numpy()
                r, g, b = trail_colors[i]
                alpha = 1.0 - (i / TRAIL_LENGTH)
                ghost_pos.append(pos)
                ghost_col.append(np.tile((r, g, b), (pos.shape[0], 1)))
                ghost_alpha.append(np.full((pos.shape[0], 1), alpha * 0.6))
                ghost_age.append(np.full((pos.shape[0], 1), trail_state['frame']))
        elif RAINBOW_MODE == BEST_MODE:
            for i, best in enumerate(best_per_scale):
                if best is not None:
                    pos = best['positions'].cpu().numpy()
                    r, g, b = rainbow_best_colors[i]
                    alpha = 0.92
                    ghost_pos.append(pos)
                    ghost_col.append(np.tile((r, g, b), (pos.shape[0], 1)))
                    ghost_alpha.append(np.full((pos.shape[0], 1), alpha))
                    ghost_age.append(np.full((pos.shape[0], 1), best['frame']))
        if ghost_pos:
            ghost_pos = np.concatenate(ghost_pos, axis=0)
            ghost_col = np.concatenate(ghost_col, axis=0)
            ghost_alpha = np.concatenate(ghost_alpha, axis=0)
            ghost_age = np.concatenate(ghost_age, axis=0)
            ghost_data = np.concatenate([ghost_pos, ghost_col, ghost_alpha, ghost_age], axis=1)
            # sort by age (last col)
            sort_idx = np.argsort(ghost_data[:,-1])
            ghost_data = ghost_data[sort_idx]
        else:
            ghost_data = np.zeros((0,8), dtype=np.float32)

        update_vbo(vbo_ghosts, ghost_data.astype(np.float32))

        # --- Draw ghosts with the same shader as nodes ---
            # --- Draw ghosts with the same shader as nodes ---
        if ghost_data.shape[0] > 0:
            # 1) Repack into pos(3), color(3), alpha, radius, ke
            N_ghosts = ghost_data.shape[0]
            pos    = ghost_data[:, 0:3]
            col    = ghost_data[:, 3:6]
            alpha  = ghost_data[:, 6:7]
            radius = np.full((N_ghosts, 1), GHOST_PEAK_RADIUS, dtype=np.float32)
            ke     = np.zeros((N_ghosts, 1), dtype=np.float32)
            ghost_node_data = np.hstack([pos, col, alpha, radius, ke]).astype(np.float32)
            update_vbo(vbo_ghosts, ghost_node_data)

            # 2) Bind shader & uniforms
            glUseProgram(shader_program)
            glUniform1f(time_loc, pygame.time.get_ticks())
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_np.flatten())

            # 3) Enable attributes exactly as for nodes
            glBindBuffer(GL_ARRAY_BUFFER, vbo_ghosts)
            stride = 9 * 4
            # in_position
            loc = glGetAttribLocation(shader_program, "in_position")
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            # in_color
            loc_c = glGetAttribLocation(shader_program, "in_color")
            glEnableVertexAttribArray(loc_c)
            glVertexAttribPointer(loc_c, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3*4))
            # in_alpha
            loc_a = glGetAttribLocation(shader_program, "in_alpha")
            glEnableVertexAttribArray(loc_a)
            glVertexAttribPointer(loc_a, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6*4))
            # in_radius
            loc_r = glGetAttribLocation(shader_program, "in_radius")
            glEnableVertexAttribArray(loc_r)
            glVertexAttribPointer(loc_r, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(7*4))
            # in_ke
            loc_k = glGetAttribLocation(shader_program, "in_ke")
            glEnableVertexAttribArray(loc_k)
            glVertexAttribPointer(loc_k, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8*4))

            # 4) Draw them all in one go
            glDrawArrays(GL_POINTS, 0, N_ghosts)

            # 5) Tear down
            glDisableVertexAttribArray(loc)
            glDisableVertexAttribArray(loc_c)
            glDisableVertexAttribArray(loc_a)
            glDisableVertexAttribArray(loc_r)
            glDisableVertexAttribArray(loc_k)
            glUseProgram(0)
        if PCA: #this will be used later for ML, inducing penalty for sequence ordering
            # --- Draw ordering strip with its own shader ---
            # Note: this is a separate shader to allow for different point sizes
            #       and to avoid mixing with the main node rendering.
            # ---------- ordering strip ------------------------------------------------
            glUseProgram(strip_prog)
            glUniform1f(loc_y,  STRIP_Y_NDC)
            glUniform1f(loc_px, STRIP_PIX_SCALE)

            glBindBuffer(GL_ARRAY_BUFFER, strip_vbo)
            stride = 9 * 4        # 9 floats, tightly packed

            glEnableVertexAttribArray(loc_ord)
            glVertexAttribPointer(loc_ord, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

            glEnableVertexAttribArray(loc_col)
            glVertexAttribPointer(loc_col, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(1*4))

            glEnableVertexAttribArray(loc_alp)
            glVertexAttribPointer(loc_alp, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(4*4))

            glEnableVertexAttribArray(loc_rad)
            glVertexAttribPointer(loc_rad, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(5*4))

            glEnableVertexAttribArray(loc_ke)
            glVertexAttribPointer(loc_ke, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6*4))

            glDisable(GL_DEPTH_TEST)
            glDrawArrays(GL_POINTS, 0, N)
            glEnable(GL_DEPTH_TEST)

            glDisableVertexAttribArray(loc_ord)
            glDisableVertexAttribArray(loc_col)
            glDisableVertexAttribArray(loc_alp)
            glDisableVertexAttribArray(loc_rad)
            glDisableVertexAttribArray(loc_ke)
            glUseProgram(0)

        # --- Draw title at top center ---
        demo_name = demo.get('name', str(menu_index))
        title_surface = title_font.render(str(demo_name), True, (220, 220, 220))
        title_rect = title_surface.get_rect(center=(WIDTH // 2, 20))
        # Switch to 2D mode for overlay
        pygame.display.set_caption(demo_name)

        pygame.display.flip()
        clock.tick(FPS)
        #pg.run_at(ordered_keys[idx][0], ordered_keys[idx][1], ordered_keys[idx][2])
        frame += 1

