
import numpy as np

def icosahedron():
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = np.array([
        [-1,  t,  0],[ 1,  t,  0],[-1, -t,  0],[ 1, -t,  0],
        [ 0, -1,  t],[ 0,  1,  t],[ 0, -1, -t],[ 0,  1, -t],
        [ t,  0, -1],[ t,  0,  1],[-t,  0, -1],[-t,  0,  1],
    ], dtype=np.float64)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=np.int32)
    verts /= np.linalg.norm(verts, axis=1)[:,None]
    return verts, faces

def subdivide(verts, faces):
    vert_list = verts.tolist()
    midpoint_cache = {}
    def midpoint(i, j):
        key = tuple(sorted((i,j)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        v = (verts[i] + verts[j]) * 0.5
        v = v / np.linalg.norm(v)
        vert_list.append(v.tolist())
        idx = len(vert_list) - 1
        midpoint_cache[key] = idx
        return idx
    new_faces = []
    for (a,b,c) in faces:
        ab = midpoint(a,b)
        bc = midpoint(b,c)
        ca = midpoint(c,a)
        new_faces += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    verts2 = np.array(vert_list, dtype=np.float64)
    faces2 = np.array(new_faces, dtype=np.int32)
    return verts2, faces2

def make_icosphere(subdiv=1, radius=0.1, center=(0,0,0)):
    v, f = icosahedron()
    for _ in range(max(0, int(subdiv))):
        v, f = subdivide(v, f)
    v = v * radius + np.array(center, dtype=np.float64)
    return v, f

def mesh_volume(verts, faces):
    V = 0.0
    for tri in faces:
        i,j,k = tri
        a,b,c = verts[i],verts[j],verts[k]
        V += np.dot(a, np.cross(b, c))
    return V/6.0

def volume_gradients(verts, faces):
    grads = np.zeros_like(verts)
    for tri in faces:
        i,j,k = tri
        a,b,c = verts[i],verts[j],verts[k]
        grads[i] += np.cross(b,c)
        grads[j] += np.cross(c,a)
        grads[k] += np.cross(a,b)
    grads /= 6.0
    return grads

def build_adjacency(faces):
    edge2tris = {}
    for t_idx,(i,j,k) in enumerate(faces):
        for e in [(i,j),(j,k),(k,i)]:
            e_sorted = tuple(sorted(e))
            edge2tris.setdefault(e_sorted, []).append((t_idx, (i,j,k)))
    edges = []
    bends = []  # (i,j,k,l)
    for (i,j), tris in edge2tris.items():
        edges.append((i,j))
        if len(tris)==2:
            (t1,(a1,b1,c1)), (t2,(a2,b2,c2)) = tris
            k = [v for v in (a1,b1,c1) if v not in (i,j)][0]
            l = [v for v in (a2,b2,c2) if v not in (i,j)][0]
            bends.append((i,j,k,l))
    return edges, bends
