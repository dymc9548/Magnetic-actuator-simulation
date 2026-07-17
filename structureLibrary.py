def structureLibrary(struc='original'):

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