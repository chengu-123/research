"""SAJO: Screw-Axis Joint Optimization with Contact Anchors.

Submodules:
    screw    - SE(3) exponential map, Plücker parameterization, manifold projection
    warp     - Differentiable trilinear warp of a 64^3 occupancy volume under SE(3)
    anchors  - Joint-free variance-based split and 26-connectivity contact anchor extraction
    init     - Dual-hypothesis (revolute + prismatic) initialization from anchors
    em       - EM optimization of (screw params, M_move) under the total energy
    bic      - Dual-model BIC joint-type selection
"""
