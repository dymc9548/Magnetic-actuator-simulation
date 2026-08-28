import copy
import numpy as np


def structureLibrary(struc='original'):
    """
    Return the predefined chain-of-shapes layout for a named structure.

    Each named structure is a hardcoded dict of shapes making up a chain,
    used as a starting configuration for the folding simulations.

    Parameters
    ----------
    struc : str, optional
        Name of the structure to build. One of: 'original', 'original_8', 'diamond',
        'six ring', 'zipper', 'two domain', 'alt corners', 'backbone',
        'end-middle', 'sym frustration', 'three domain', 'alt dipole',
        'dumbbell', 'greedy trap'. Defaults to 'original'.

    Returns
    -------
    shapes : dict
        Maps 'shape N' -> [shape_type, size, offset, rotation, patches].
        - shape_type (str): shape identifier, e.g. 's' for square.
        - size (int/float): shape size.
        - offset (int/float): placement offset of this shape from the
          previous one in the chain.
        - rotation (int/float): initial rotation angle.
        - patches (dict): maps 'patch N' -> [corner, size, angle], where
          corner is one of 'top left', 'top right', 'bottom left',
          'bottom right', identifying where on the shape the patch sits.
          May be empty if the shape has no patches.
    """

    if struc == 'original':
        shapes = {
            'shape 1': ['s', 10, 0, 0, 
                        {'patch 1': ['top right', 4, 0]}], 
            'shape 2':['s', 10, 6, 0, 
                        {'patch 1': ['top left', 4, 0], 
                         'patch 2': ['bottom right', 4, 0]}], 
            'shape 3': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom left', 4, 0]}], 
            'shape 4': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom left', 4, 0]}], 
            'shape 5': ['s', 10, 6, 0, 
                        {'patch 1': ['top left', 4, 0]}], 
            'shape 6': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 4, 0]}], 
            'shape 7': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 4, 0]}], 
            'shape 8': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 4, 0]}], 
            'shape 9': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 4, 0]}], 
            'shape 10': ['s', 10, 6, 0, 
                         {'patch 1': ['bottom right', 4, 0]}], 
            'shape 11': ['s', 10, 6, 0, 
                         {'patch 1': ['bottom right', 4, 0]}]}

    elif struc == 'original_8':
        shapes = {
            'shape 1': ['s', 10, 0, 0, 
                        {'patch 1': ['top right', 8, 1]}], 
            'shape 2':['s', 10, 6, 0, 
                        {'patch 1': ['top left', 8, 1], 
                         'patch 2': ['bottom right', 8, 1]}], 
            'shape 3': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom left', 8, 1]}], 
            'shape 4': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom left', 8, 1]}], 
            'shape 5': ['s', 10, 6, 0, 
                        {'patch 1': ['top left', 8, 1]}], 
            'shape 6': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 8, 1]}], 
            'shape 7': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 8, 1]}], 
            'shape 8': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 8, 1]}], 
            'shape 9': ['s', 10, 6, 0, 
                        {'patch 1': ['bottom right', 8, 1]}], 
            'shape 10': ['s', 10, 6, 0, 
                         {'patch 1': ['bottom right', 8, 1]}], 
            'shape 11': ['s', 10, 6, 0, 
                         {'patch 1': ['bottom right', 8, 1]}]}

    elif struc == 'diamond':
        shapes = {
            'shape 1': ['s', 10, 0, 0,
                        {'patch 1': ['top right', 4, 0]}],

            'shape 2': ['s', 10, 6, 0,
                        {'patch 1': ['top left', 4, 0],
                        'patch 2': ['bottom right', 4, 0]}],

            'shape 3': ['s', 10, 6, 0,
                        {'patch 1': ['bottom left', 4, 0],
                        'patch 2': ['top right', 4, 0]}],

            'shape 4': ['s', 10, 6, 0,
                        {'patch 1': ['bottom left', 4, 0]}]
        }

    elif struc == 'six ring':
        shapes = {
            'shape 1': ['s', 10, 0, 0,
                        {'patch 1': ['top right', 4, 0]}],

            'shape 2': ['s', 10, 6, 0,
                        {'patch 1': ['top left', 4, 0],
                        'patch 2': ['top right', 4, 0]}],

            'shape 3': ['s', 10, 6, 0,
                        {'patch 1': ['top left', 4, 0],
                        'patch 2': ['bottom right', 4, 0]}],

            'shape 4': ['s', 10, 6, 0,
                        {'patch 1': ['bottom left', 4, 0],
                        'patch 2': ['bottom right', 4, 0]}],

            'shape 5': ['s', 10, 6, 0,
                        {'patch 1': ['bottom left', 4, 0],
                        'patch 2': ['top right', 4, 0]}],

            'shape 6': ['s', 10, 6, 0,
                        {'patch 1': ['top left', 4, 0]}]
        }

    elif struc == 'zipper':
        shapes = {
            'shape 1': ['s',10,0,0,{'patch 1':['top right',4,0]}],

            'shape 2': ['s',10,6,0,
                        {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 3': ['s',10,6,0,
                        {'patch 1':['bottom left',4,0],
                        'patch 2':['top right',4,0]}],

            'shape 4': ['s',10,6,0,
                        {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 5': ['s',10,6,0,
                        {'patch 1':['bottom left',4,0],
                        'patch 2':['top right',4,0]}],

            'shape 6': ['s',10,6,0,
                        {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 7': ['s',10,6,0,
                        {'patch 1':['bottom left',4,0],
                        'patch 2':['top right',4,0]}],

            'shape 8': ['s',10,6,0,
                        {'patch 1':['top left',4,0]}]
        }

    elif struc == "two domain":
        shapes = {
            'shape 1':['s',10,0,0,
                    {'patch 1':['top right',4,0]}],

            'shape 2':['s',10,6,0,
                    {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 3':['s',10,6,0,
                    {'patch 1':['bottom left',4,0],
                        'patch 2':['top right',4,0]}],

            'shape 4':['s',10,6,0,
                    {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 5':['s',10,6,0,
                    {'patch 1':['bottom left',4,0]}],

            'shape 6':['s',10,6,0,
                    {'patch 1':['top right',4,0]}],

            'shape 7':['s',10,6,0,
                    {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 8':['s',10,6,0,
                    {'patch 1':['bottom left',4,0],
                        'patch 2':['top right',4,0]}],

            'shape 9':['s',10,6,0,
                    {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

            'shape 10':['s',10,6,0,
                        {'patch 1':['bottom left',4,0]}],

            'shape 11':['s',10,6,0,
                        {'patch 1':['top left',4,0]}]
        }
        
    elif struc == 'alt corners':
        shapes = {
            'shape 1':['s',10,0,0,
                    {'patch 1':['top right',4,0]}],

            'shape 2':['s',10,6,0,
                    {'patch 1':['bottom right',4,0]}],

            'shape 3':['s',10,6,0,
                    {'patch 1':['top left',4,0]}],

            'shape 4':['s',10,6,0,
                    {'patch 1':['bottom left',4,0]}],

            'shape 5':['s',10,6,0,
                    {'patch 1':['top right',4,0]}],

            'shape 6':['s',10,6,0,
                    {'patch 1':['bottom right',4,0]}],

            'shape 7':['s',10,6,0,
                    {'patch 1':['top left',4,0]}],

            'shape 8':['s',10,6,0,
                    {'patch 1':['bottom left',4,0]}]
        }    
    elif struc == 'backbone':
        shapes = {
        'shape 1':['s',10,0,0,{'patch 1':['top right',4,0]}],

        'shape 2':['s',10,6,0,{'patch 1':['top left',4,0]}],

        'shape 3':['s',10,6,0,{}],

        'shape 4':['s',10,6,0,{}],

        'shape 5':['s',10,6,0,{}],

        'shape 6':['s',10,6,0,{}],

        'shape 7':['s',10,6,0,{}],

        'shape 8':['s',10,6,0,{'patch 1':['bottom right',4,0]}],

        'shape 9':['s',10,6,0,{'patch 1':['bottom left',4,0]}]
        }
        
    elif struc == 'end-middle':
        shapes = {
        'shape 1':['s',10,0,0,
                {'patch 1':['top right',4,0]}],

        'shape 2':['s',10,6,0,{}],

        'shape 3':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 4':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 5':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 6':['s',10,6,0,{}],

        'shape 7':['s',10,6,0,
                {'patch 1':['top left',4,0]}],

        'shape 8':['s',10,6,0,{}],

        'shape 9':['s',10,6,0,
                {'patch 1':['bottom right',4,0]}]
        }

    elif struc == 'sym frustration':
        shapes = {
        'shape 1':['s',10,0,0,
                {'patch 1':['top right',4,0]}],

        'shape 2':['s',10,6,0,
                {'patch 1':['bottom right',4,0]}],

        'shape 3':['s',10,6,0,
                {'patch 1':['top left',4,0]}],

        'shape 4':['s',10,6,0,{}],

        'shape 5':['s',10,6,0,
                {'patch 1':['bottom right',4,0]}],

        'shape 6':['s',10,6,0,
                {'patch 1':['top left',4,0]}],

        'shape 7':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}]
        }

    elif struc == 'three domain':
        shapes = {
        'shape 1':['s',10,0,0,
                {'patch 1':['top right',4,0]}],

        'shape 2':['s',10,6,0,
                {'patch 1':['top left',4,0]}],

        'shape 3':['s',10,6,0,{}],

        'shape 4':['s',10,6,0,
                {'patch 1':['bottom right',4,0]}],

        'shape 5':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 6':['s',10,6,0,{}],

        'shape 7':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 8':['s',10,6,0,
                {'patch 1':['top left',4,0]}],

        'shape 9':['s',10,6,0,{}],

        'shape 10':['s',10,6,0,
                        {'patch 1':['bottom right',4,0]}],

        'shape 11':['s',10,6,0,
                        {'patch 1':['bottom left',4,0]}]
        }

    elif struc == 'alt dipole':
        shapes = {
        'shape 1':['s',10,0,0,
                {'patch 1':['top right',4,0]}],

        'shape 2':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 3':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 4':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 5':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 6':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 7':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 8':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 9':['s',10,6,0,
                {'patch 1':['top right',4,0]}]
        }

    elif struc == 'dumbbell':
        shapes = {
        'shape 1':['s',10,0,0,
                {'patch 1':['top right',4,0],
                        'patch 2':['bottom left',4,0]}],

        'shape 2':['s',10,6,0,
                {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}],

        'shape 3':['s',10,6,0,{}],

        'shape 4':['s',10,6,0,{}],

        'shape 5':['s',10,6,0,{}],

        'shape 6':['s',10,6,0,{}],

        'shape 7':['s',10,6,0,{}],

        'shape 8':['s',10,6,0,
                {'patch 1':['top right',4,0],
                        'patch 2':['bottom left',4,0]}],

        'shape 9':['s',10,6,0,
                {'patch 1':['top left',4,0],
                        'patch 2':['bottom right',4,0]}]
        }

    elif struc == 'greedy trap':
        shapes = {
        'shape 1':['s',10,0,0,
                {'patch 1':['top right',4,0]}],

        'shape 2':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 3':['s',10,6,0,{}],

        'shape 4':['s',10,6,0,{}],

        'shape 5':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 6':['s',10,6,0,
                {'patch 1':['bottom left',4,0]}],

        'shape 7':['s',10,6,0,{}],

        'shape 8':['s',10,6,0,{}],

        'shape 9':['s',10,6,0,
                {'patch 1':['top right',4,0]}],

        'shape 10':['s',10,6,0,
                        {'patch 1':['bottom left',4,0]}]
        }

    # elif struc == :
    
    # elif struc == :

    return shapes


def _sample_hinge_offset(bar_radius, gap, steric_clearance, rng):
    '''Samples one point (dx, dy) uniformly at random: the hinge/bar center's position relative to
    where the arm (rigidly attached to the right/downstream shape) nominally expects it to be.

    The bar (radius bar_radius) sits inside the arm's bore; gap is the bore's radius, so the bar's
    center can wander up to (gap - bar_radius) off-axis before the bar's own edge touches the
    bore's inner wall -- that distance bounds a disk of allowed (dx, dy) around the nominal,
    concentric position.

    In the -x direction (toward the previous/left shape, which the bar is rigidly attached to),
    steric hindrance from that shape caps how far the hinge can shift that way to steric_clearance,
    tighter than the (gap - bar_radius) bound that applies in every other direction. Sampled via
    rejection so the result is uniform over that (circle-minus-a-cap) region.'''
    radius = gap - bar_radius
    x_min = -steric_clearance

    while True:
        theta = rng.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(rng.uniform(0, 1))
        x, y = r * np.cos(theta), r * np.sin(theta)
        if x >= x_min:
            return x, y


def perturb_structure(shapes, patch_offset_std=0.0, patch_offset_mean=0.0,
                       hinge_bar_radius=0.0, hinge_gap=0.0, hinge_steric_clearance=0.0,
                       hinge_left_offset=None, hinge_rotation_std=0.0, hinge_rotation_mean=0.0,
                       rng=None):
    '''
    Applies fabrication-uncertainty noise to a structure dict from structureLibrary().
    All perturbations default to off, so perturb_structure(shapes) with no other
    arguments returns a structure that generates identically to the input.

    Inputs:
        shapes: structure dict, as returned by structureLibrary()
        patch_offset_std, patch_offset_mean: (float) std/mean of the Gaussian noise added to
            each patch's offset from its corner. The result is clamped to
            [0, shape size - patch size] so the patch can't run off the shape's edge.
        hinge_bar_radius: (float) radius of the rigid bar half of the hinge. The bar is bolted to
            the *previous* shape at hinge_left_offset, and stays fixed there regardless of this
            shape's own fabrication slop -- like a monkey bar bolted to it. 0 disables hinge
            position perturbation (hinge_gap is also required to enable it).
        hinge_gap: (float) radius of the bore in the arm (rigidly attached to *this* shape) that
            the bar sits inside of. Since the bar has to stay inside that bore, its center -- and
            so this shape's whole position relative to it -- can wobble by up to (hinge_gap -
            hinge_bar_radius) off the nominal, concentric alignment before the bar's edge touches
            the bore wall. This shape (and everything downstream of it) is offset from where the
            arm nominally expects the fixed bar center to be, by a point sampled uniformly over
            that disk. 0 disables this (hinge_bar_radius is also required).
        hinge_steric_clearance: (float) how far, at most, this shape is allowed to shift *toward*
            the previous shape (i.e. the bar/hinge gets closer than its nominal, concentric
            position) before steric hindrance from that shape stops it. Tighter than the
            (hinge_gap - hinge_bar_radius) bound that applies in every other direction.
        hinge_left_offset: (float, optional) nominal distance from the previous shape's actual
            edge to the bar/hinge center (e.g. 3.8 for a bar bolted well short of the halfway
            point). None (the default) keeps the hinge at the halfway point, nominal_spacing / 2,
            same as if this were a plain scalar offset. Applies even if hinge_gap is 0, so it can
            also be used to fix an asymmetric nominal hinge position with no added wobble.
        hinge_rotation_std, hinge_rotation_mean: (float, degrees) std/mean of the Gaussian noise
            added to each hinge's initial angle, via the shape dict's rotation field. A std of 0
            (the default) leaves hinges at the nominal flat (180 deg) starting angle.
        rng: numpy.random.Generator, optional. A fresh default_rng() is used if omitted; pass a
            seeded Generator for reproducible perturbed structures.

    Return:
        dict: a deep copy of shapes with the requested noise applied
    '''
    if rng is None:
        rng = np.random.default_rng()

    perturbed = copy.deepcopy(shapes)
    keys = list(perturbed.keys())

    for i, key in enumerate(keys):
        shape_type, size, offset, rotation, patches = perturbed[key]

        if patch_offset_std > 0 or patch_offset_mean != 0:
            for patch in patches.values():
                corner, plength, poffset = patch
                poffset += rng.normal(patch_offset_mean, patch_offset_std)
                patch[2] = min(max(poffset, 0), size - plength)

        if i > 0: # shape 1 has no incoming hinge, so hinge perturbations don't apply to it
            left_offset = hinge_left_offset if hinge_left_offset is not None else offset / 2

            if hinge_gap > 0:
                dx, dy = _sample_hinge_offset(hinge_bar_radius, hinge_gap, hinge_steric_clearance, rng)
                perturbed[key][2] = (offset + dx, dy, offset, left_offset) # (actual spacing, actual vertical shift, nominal spacing, hinge offset from previous shape) -- generate() places the hinge from the nominal spacing/hinge offset only, and this shape from the actual values
            elif hinge_left_offset is not None:
                perturbed[key][2] = (offset, 0.0, offset, left_offset)

            if hinge_rotation_std > 0 or hinge_rotation_mean != 0:
                perturbed[key][3] = rotation + rng.normal(hinge_rotation_mean, hinge_rotation_std)

    return perturbed