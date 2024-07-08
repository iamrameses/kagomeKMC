# src/plotter.py

import freud
from matplotlib.patches import Polygon
import numpy as np


def get_disclinations(L, mxy):
    # Compute the Voronoi tessellation and the hexatic order parameter
    xdim, ydim = L.boxsize
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    vor = freud.locality.Voronoi()
    psi6 = freud.order.Hexatic(k=6, weighted=False)
    points = np.hstack((mxy-L.center_xy, np.zeros((mxy.shape[0], 1))))
    vor.compute(system=(box, points))
    psi6.compute(system=(box, points), neighbors=vor.nlist)
    psi6_k = psi6.particle_order
    psi6_avg = np.mean(np.abs(psi6_k))
    nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
    # Get the patches for the Voronoi tessellation
    patches = []
    for polytope in vor.polytopes:
        poly = Polygon(polytope[:,:2]+L.center_xy, closed=True, facecolor='r')
        patches.append(poly)
    return patches, nsides, psi6_avg

def get_global_orientational_order(L, mxy, box=None, vor=None, psi6=None):
    # Compute the Voronoi tessellation and the hexatic order parameter
    xdim, ydim = L.boxsize
    if box is None:
        box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    if vor is None:
        vor = freud.locality.Voronoi()
    if psi6 is None:
        psi6 = freud.order.Hexatic(k=6, weighted=False)
    ndims = len(mxy.shape)
    if ndims == 3:
        gboops = np.zeros(mxy.shape[0])
        for i, rxy in enumerate(mxy):
            points = np.hstack((rxy-L.center_xy, np.zeros((rxy.shape[0], 1))))
            vor.compute(system=(box, points))
            psi6.compute(system=(box, points), neighbors=vor.nlist)
            psi6_k = psi6.particle_order
            gboops[i] = np.mean(np.abs(psi6_k))
            # psi6_phase = np.abs(np.angle(psi6.particle_order)) 
    elif ndims == 2:
        points = np.hstack((mxy-L.center_xy, np.zeros((mxy.shape[0], 1))))
        vor.compute(system=(box, points))
        psi6.compute(system=(box, points), neighbors=vor.nlist)
        psi6_k = psi6.particle_order
        gboops = np.mean(np.abs(psi6_k))
    return gboops