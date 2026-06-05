"""
Small numpy geometry primitives shared by the fast (torch-less) pipeline.

Ports of the three operations used by LASErMPNN's hydrogen placement:
  * kabsch()          rigid superposition (compute_alignment_matrices)
  * extend_coordinate() NeRF placement of a 4th atom from an internal coordinate
  * rotate_about_axis() Rodrigues rotation (used by the dihedral sweeps)

All functions operate on plain numpy arrays in Angstroms.  No torch.
"""
from __future__ import annotations

import numpy as np


def kabsch(mobile: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotation/translation that best superposes ``mobile`` onto ``fixed``.

    Both are (N, 3).  Returns (R, mobile_com, fixed_com) such that
    ``apply_transform(coords, R, mobile_com, fixed_com)`` maps mobile-frame
    coordinates onto the fixed frame.  Mirrors LASErMPNN
    compute_alignment_matrices (SVD with a determinant-sign fix).
    """
    mobile_com = mobile.mean(axis=0)
    fixed_com = fixed.mean(axis=0)
    m = mobile - mobile_com
    f = fixed - fixed_com
    cov = m.T @ f
    u, _, wt = np.linalg.svd(cov)
    r = u @ wt
    if np.linalg.det(r) < 0.0:
        wt = wt.copy()
        wt[-1] *= -1
        r = u @ wt
    return r, mobile_com, fixed_com


def apply_transform(coords: np.ndarray, r: np.ndarray,
                    mobile_com: np.ndarray, fixed_com: np.ndarray) -> np.ndarray:
    """Apply a (R, mobile_com, fixed_com) transform from :func:`kabsch`."""
    return (coords - mobile_com) @ r + fixed_com


def extend_coordinate(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      bond_length: float, bond_angle_rad: float,
                      dihedral_rad: float) -> np.ndarray:
    """
    Place a 4th atom D from three reference atoms A-B-C (NeRF / SN-NeRF).

    Geometry: |C-D| = bond_length, angle B-C-D = bond_angle_rad,
    dihedral A-B-C-D = dihedral_rad.  Port of LASErMPNN extend_coordinates.
    """
    eps = 1e-9
    bc = c - b
    bc = bc / (np.linalg.norm(bc) + eps)
    ba = np.cross(b - a, bc)
    ba = ba / (np.linalg.norm(ba) + eps)
    m = np.cross(ba, bc)  # third orthonormal axis
    d1 = bond_length * np.cos(bond_angle_rad)
    d2 = bond_length * np.sin(bond_angle_rad) * np.cos(dihedral_rad)
    d3 = bond_length * np.sin(bond_angle_rad) * np.sin(dihedral_rad)
    return c - bc * d1 + m * d2 + ba * d3


def rotate_about_axis(points: np.ndarray, origin: np.ndarray,
                      axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate ``points`` (..., 3) by ``theta`` radians about the line through
    ``origin`` with direction ``axis`` (Rodrigues formula).
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = points - origin
    proj = np.asarray(v @ axis)[..., None] * axis  # (axis . v) axis, shape-preserving
    return (origin
            + v * np.cos(theta)
            + np.cross(axis, v) * np.sin(theta)
            + proj * (1.0 - np.cos(theta)))


def internal_coords(x: np.ndarray, neighbor: np.ndarray,
                    next_neighbor: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """
    Return (bond_length, bond_angle_rad) for an X-H bond given the heavy-atom
    neighbour chain next_neighbor-neighbor-X and the ideal H position.

    bond_length = |X-H|, bond_angle = angle(neighbor-X-H).
    """
    bl = float(np.linalg.norm(h - x))
    v1 = neighbor - x
    v2 = h - x
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return bl, float(np.arccos(np.clip(cos, -1.0, 1.0)))


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed dihedral A-B-C-D in radians."""
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))
