"""Conservative closed/outward proof at Studio float32 mesh precision."""
import numpy as np
from itertools import product

# Radius-unit coordinates travel as float32. A 1e-7 grid welds duplicate
# parametric seam/pole vertices; it does not alter the transported geometry.
# Features smaller than this admission precision are unsupported, not repaired.
TOPOLOGY_GRID = 1e-7


def _weld(vertices):
    # Neighbour cells avoid a rounding-grid boundary splitting two copies
    # of one seam vertex. Only existing coordinate copies are identified.
    points,inverse=np.unique(vertices,axis=0,return_inverse=True)
    parent=np.arange(len(points));cells={}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for i,p in enumerate(points):
        cell=tuple(np.floor(p/TOPOLOGY_GRID).astype(np.int64))
        for offset in product((-1,0,1),repeat=3):
            key=tuple(cell[k]+offset[k] for k in range(3))
            for j in cells.get(key,()):
                if np.dot(p-points[j],p-points[j])<=TOPOLOGY_GRID**2:
                    parent[root(i)]=root(j)
        cells.setdefault(cell,[]).append(i)
    ids=np.array([root(i) for i in range(len(points))])
    return points,ids[inverse]


def require_closed_outward(triangles):
    tri=np.asarray(triangles,np.float64)
    points,inverse=_weld(tri.reshape(-1,3))
    faces=inverse.reshape(-1,3)
    valid=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,2]!=faces[:,0])
    faces=faces[valid];source=tri[valid]
    if len(faces)<4:raise ValueError('optical volume has no closed surface at mesh precision')
    directed=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    canonical=np.sort(directed,axis=1)
    _,edge_ids,counts=np.unique(canonical,axis=0,return_inverse=True,return_counts=True)
    signs=np.where(directed[:,0]<directed[:,1],1,-1)
    if np.any(counts!=2) or np.any(np.bincount(edge_ids,weights=signs)!=0):
        raise ValueError('optical volume must be closed with consistent winding')
    # Connected components cannot hide an inward shell behind a larger
    # outward one. Work over vertices only after every edge is manifold.
    parent=np.arange(len(points))
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in canonical:
        ra,rb=root(a),root(b)
        if ra!=rb:parent[rb]=ra
    labels=np.array([root(f[0]) for f in faces])
    six_volumes=np.einsum('ij,ij->i',source[:,0],np.cross(source[:,1],source[:,2]))
    for component in np.unique(labels):
        if np.sum(six_volumes[labels==component])<=0:
            raise ValueError('optical volume winding must face outward')

def centered_insphere_radius(triangles):
    """Conservative radius about origin, verified inside a closed oriented shell.

    Used only for generated dice admission; does not claim exact arbitrary
    inclusion/host intersection. Every inclusion must fit strictly in this ball.
    """
    tri=np.asarray(triangles,np.float64);require_closed_outward(tri)
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    la,lb,lc=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)
    num=np.einsum('ij,ij->i',a,np.cross(b,c))
    den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    winding=np.sum(2*np.arctan2(num,den))
    if not np.isfinite(winding) or abs(winding-4*np.pi)>1e-4:
        raise ValueError('die host must enclose its origin')
    normal=np.cross(b-a,c-a);n2=np.einsum('ij,ij->i',normal,normal)
    valid=n2>1e-28
    normal=normal[valid];a,b,c=a[valid],b[valid],c[valid];n2=n2[valid]
    signed=np.einsum('ij,ij->i',a,normal)
    projection=normal*(signed/n2)[:,None]
    inside=np.ones(len(a),bool)
    distances=[]
    for u,v in ((a,b),(b,c),(c,a)):
        edge=v-u
        inside &= np.einsum('ij,ij->i',np.cross(edge,projection-u),normal)>=-1e-18
        t=np.clip(-np.einsum('ij,ij->i',u,edge)/np.einsum('ij,ij->i',edge,edge),0,1)
        distances.append(np.linalg.norm(u+t[:,None]*edge,axis=1))
    distance=np.minimum.reduce(distances)
    distance[inside]=np.abs(signed[inside])/np.sqrt(n2[inside])
    radius=float(np.min(distance))
    if not np.isfinite(radius) or radius<=1e-7:raise ValueError('die host has no resolved centered interior')
    return radius
