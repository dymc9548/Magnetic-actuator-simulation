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