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


def _sample_disk(radius, rng):
    '''Samples one point uniformly at random from within a disk of the given radius, centered on the origin.
    The x-component is folded to be non-negative, so the actual hinge spacing (nominal + dx) can only grow
    or stay the same relative to nominal, never shrink.'''
    theta = rng.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(rng.uniform(0, 1))
    return abs(r * np.cos(theta)), r * np.sin(theta)


def perturb_structure(shapes, patch_offset_std=0.0, patch_offset_mean=0.0,
                       hinge_radius=0.0, hinge_rotation_std=0.0, hinge_rotation_mean=0.0,
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
        hinge_radius: (float) radius of the disk within which each shape's actual placement
            may land relative to its hinge point. The hinge/pivot itself stays fixed to the
            *previous* shape (like a monkey bar bolted to it); this shape (and everything
            downstream of it) is offset from that fixed pivot by a random point sampled
            uniformly over the disk's area -- shifting it left/right and up/down. 0 disables this.
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
            if hinge_radius > 0:
                dx, dy = _sample_disk(hinge_radius, rng)
                perturbed[key][2] = (offset + dx, dy, offset) # (actual spacing, actual vertical shift, nominal spacing) -- generate() places the hinge from nominal spacing only, and this shape from the actual values

            if hinge_rotation_std > 0 or hinge_rotation_mean != 0:
                perturbed[key][3] = rotation + rng.normal(hinge_rotation_mean, hinge_rotation_std)

    return perturbed