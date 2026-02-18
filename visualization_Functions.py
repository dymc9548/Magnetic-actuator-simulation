# Functions to help with the initial visualization of the system

import numpy as np
import matplotlib.pyplot as plt
import shapely as sp
from shapely.ops import polygonize
import copy


def generate(shapes):
    '''Generates the desired shapes in 2D space. The shapes have no overlap and are an idealized version of the structure.
    Inputs:
        shapes: Dictionary object where first entry is the shape, second entry is shape length, third entry is shape spacing from previous, fourth entry is uncertainty in spacing,
                and fifth entry is a dictionary of patches (where the first entry is patch location, secondis patch length, third is patch offset from edge)
            example: shapes = {'shape 1': ['s', 10, 6, 0, {'patch 1': ['top right', 4, 0]}], 'shape 2':['s', 10, 6, 0, {'patch 1': ['top left', 4, 0], 'patch 2': ['bottom right', 4, 0]}], 'shape 3': ['s', 10, 6, 0, {'patch 1': ['bottom left', 4, 0]}]}
    Return:
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        hinge_loc: 2x(n-1) array of x and y points that contain hinge locations. n is the number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        patch_num: n-dim list of number of patches, where each index is a shape (n shapes)'''
    
    #Separate the origin into x and y components. Note that the origin will be updated with the addition of each shape
    origin = [0,0] #the origin is at 0,0
    xi = origin[0] #Initialize the x-coordiante of the initial point for each shape
    yi = origin[1] #Initialize the y-coordinate of the initial point for each shape

    indexvec = ['shape ']*len(shapes) #Create vector to hold dictionary indices
    shape_arr = np.zeros((2,2)) #Initialize shape_arr
    linelist = [] #initialize linelist
    patch_num = [] #initialize patch number list

    #For loop  goes through each shape in the shape dictionary
    for i in range(len(shapes)):
        indexvec[i] += str(i+1) #Generate the index for each shape

    #For loop  goes through each shape in the shape dictionary
    for i in range(len(shapes)):
        
        l = shapes[indexvec[i]][1] #Store the edge length of a cube as a single variable to be referenced for creating all shapes
        d = shapes[indexvec[i]][2] #Store the unit spacing between cubes as a single variable
        patches = shapes[indexvec[i]][4] #Store the patches dictionary
        patch_indexvec = ['patch ']*len(patches) #Create vector to hold dictionary indices
        patch_num.append(len(patches)) #append the number of patches to the patch list

        #Store the hinge location
        if i > 0: #As long as we aren't on the first shape
            hx = xi - d/2 #hinge x-coordinate is halfway back to the first shape
            hy = yi + l/2 #hinge y-coordinate is halfway up the shape
            if i == 1:
                hinge_loc = np.array([hx, hy])[:,None] #make the hinge array a column vector for easy rotation
            else:
                hinge_loc = np.hstack((hinge_loc, np.array(([hx],[hy]))))

        #For loop goes through each patch in this shape's patch dictionary
        for j in range(len(patches)):
            patch_indexvec[j] += str(j+1) #Generate the index for each patch

        #Determine if the shape is a cube (or square in 2D).
        if shapes[indexvec[i]][0] == 's':
            
            linelist.append(4) #Append 4 sides to linelist

            xf = xi + l #Determine the final x-coordinate of the shape
            yf = yi + l

            top = np.array(([xi,xf],[yf,yf]))
            right = np.array(([xf,xf],[yi,yf]))
            left = np.array(([xi,xi],[yi,yf]))
            bottom = np.array(([xi,xf],[yi,yi]))

            if i == 0:
                shape_arr[:,:] = bottom #the first entry into shape_arr is a bottom. This is needed for intialization
            else:
                shape_arr = np.hstack((shape_arr,bottom)) #After the first round, append each bottom to shape_arr
                
            shape_arr = np.hstack((shape_arr,left,top,right)) #Append all other sides to shape_arr
            
            #For loop goes through each patch in this shape's patch dictionary
            for j in range(len(patches)):

                plength = patches[patch_indexvec[j]][1] #store the patch length
                poffset = patches[patch_indexvec[j]][2] #store the patch offset

                #Set up the locations of the patches
                if patches[patch_indexvec[j]][0] == 'top right':
                    pxi = xi + l - plength - poffset
                    pxf = pxi + plength
                    pyi = l
                    pyf = l

                elif patches[patch_indexvec[j]][0] == 'top left':
                    pxi = xi + poffset
                    pxf = pxi + plength
                    pyi = l
                    pyf = l
                    
                elif patches[patch_indexvec[j]][0] == 'bottom right': 
                    pxi = xi + l - plength - poffset
                    pxf = pxi + plength
                    pyi = yi
                    pyf = yi

                elif patches[patch_indexvec[j]][0] == 'bottom left':
                    pxi = xi + poffset
                    pxf = pxi + plength
                    pyi = yi
                    pyf = yi

                #else:
                    # maybe figure out a way to raise an error here

                #Add the patch locations to the patch array list
                if (i == 0) and (j == 0): #if this is the first patch on the first shape
                    patch_arr = np.array([float(pxi),float(pyi)])[:,None] #make the patch array a column vector for easy rotation
                    patch_arr = np.hstack((patch_arr, np.array(([float(pxf)],[float(pyf)]))))
                else: 
                    patch_arr = np.hstack((patch_arr, np.array(([float(pxi)],[float(pyi)]))))
                    patch_arr = np.hstack((patch_arr, np.array(([float(pxf)],[float(pyf)]))))
                
            if i < len(shapes)-1: #As long as it's before the last shape
                next_d = shapes[indexvec[i+1]][2] #Store the unit spacing between cubes as a single variable
                xi = xf+next_d #Update the origin position for generating the next shape

        #else:
            #add code for other shapes here

    hinge_vec = np.ones(len(shapes)-1)*180 #Generate vector of hinges. These represent the interdipolar angle between each of the cubes at
                                           #a hinge point.

    return hinge_vec,hinge_loc,shape_arr,linelist,patch_arr,patch_num

    
def shapeplots(shape_arr, linelist, blocking = False, title = '', show = True, bounds = '', mag_vecs = []):
    '''Plots all of the shapes in 2D space
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist:List of integers describing the number of sides in each shape
        blocking: (optional) boolean describes if plot appearing blocks further code or not
        title: (optional) string title of plot
        bounds: (optional) list of x and y bounds for the plot [xmin xmax ymin ymax]
        mag_vecs: (optinal) use patch_arr as the input for this to plot the magnetic vectors over the patches'''

    plt.figure() #Initialize figure

    shape_start = 0 #Initialize index where the shapes start
    shape_end = 0 #initialize index where the shapes end

    for i in range(len(linelist)): #for each shape
        shape_end += linelist[i]*2 #The new index is twice the number of sides in the shape
        plt.plot(shape_arr[0,shape_start:shape_end],shape_arr[1,shape_start:shape_end],'k') #Plot all values of shape array in black
        shape_start = shape_end #The new start is the previous ending

    if len(mag_vecs) != 0: #If plotting magnetization vectors is desired
        #Plot each magnetization vector as an arrow
        for i in range(0,np.shape(mag_vecs)[1],2):
            plt.arrow(mag_vecs[0,i],mag_vecs[1,i],mag_vecs[0,i+1]-mag_vecs[0,i],mag_vecs[1,i+1]-mag_vecs[1,i], color = 'r', linewidth = 3, head_width = 1, length_includes_head = True)
    
    plt.axis('square') #Fix the axis

    if title != '': #Add title if desired
        plt.title(title)
    if bounds != '': #Add bounds if desired
        plt.axis(bounds)
    if show == True: #Block the rest of the code if desired
        plt.show(block = blocking) #Show plot


def count_shapes(shape_arr):
    '''This functions uses the shapely library to count the number of shapes in a structure
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
    Outputs:
        polycount: number of shapes counted
    '''

    polygons = list(polygonize(sp.unary_union(sp.LineString(shape_arr.T)))) #Turn all the points into shapely polygons
    polycount = len(polygons) #The length of polygons is the number of polygons
    
    return polycount


def check_overlap(shape_arr, polycount):
    '''This function checks if any structures overlap each other
    Inputs:
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        polycount: Number of shapes calculated
                  
    Outputs:
        Boolean value
            True for overlap
            False for no overlap
    '''

    #use shapely library to create a LineString object. This object can be truned into a unary_union which can be turned into separate polygons using shapely.ops polygonize
    polygons = list(polygonize(sp.unary_union(sp.LineString(shape_arr.T)))) #Create list of shapely polygons

    if len(polygons)!= polycount: #If there are more polygons created that the number of shapes we are using, the shapes must be overlapping
        return True #Overlap
    else:
        return False #No overlap


def translate_to_origin(patch_arr_init, shape_arr_init, hinge_loc_init, hingechoice):
    '''function translates first point of shape to the right of the hinge to the origin. Necessary for simple rotation
    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patches. n is number of patches
        shape_ar_init: 2x(2m*s) array of x and y points that describe the lines enclosing each shape. m is number of shapes. s is number of sides on that shape
        hinge_loc: 2x(m-1) array of x and y points that contain hinge locations. m is the number of shapes
        hingechoice: Chosen hinge to revolve around (0 is first hinge)
        Collin code has: sym: (optional) Boolean indicating whether or not the translation is for the symmetry function (may need to update when doing sym score)
    Outputs:
        new_line_arr: translated line_arr
        new_shape_arr: translated shape_arr
    '''
    #Create a deepcopy of both arrays (necessary to prevent modifying original when performing trial runs in Monte Carlo)
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_loc = copy.deepcopy(hinge_loc_init)

    xy_trans = hinge_loc_init[:,hingechoice][:,None] #Determine how far to translate each shape in order to place first point past hinge on the origin

    patch_arr -= xy_trans #translate line_arr
    shape_arr -= xy_trans #translate shape_arr
    hinge_loc -= xy_trans #translate hinge_loc
    
    return patch_arr,shape_arr,hinge_loc


def rotate(patch_arr_init, shape_arr_init, linelist, hinge_vec_init, hinge_loc_init, hingechoice, angle, patch_num):
    '''This function rotates a structure. All points to the right of the origin are rotated by multiplying by teh rotation matrix
    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        hingechoice: integer chosen hinge about which to revolve
        angle: float chosen angly by which to revolve
        patch_num: n-dim list of patches per shape, where n is number of shapes
    #Outputs:
        new patch-arr: rotated patch_arr
        new_shape_arr: rotated shape_arr
        new_hinge_vec: rotated hinge_vec
    '''
    hinge_vec = copy.deepcopy(hinge_vec_init)
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_loc = copy.deepcopy(hinge_loc_init)
    #Create a deepcopy of four arrays to be modified (necessary to prevent modifying original when performing trial runs in Monte Carlo)
    hinge_vec[hingechoice] += angle #Modify hinge_vec by adding the angle
    
    angle = angle*np.pi/180 #convert angle to radians
    #Rotation matrix: [cos -sin]
    #                 [sin  cos]
    rotation_matrix =np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) #Build the rotation matrix for that angle
    
    patches_left = np.sum(patch_num[:hingechoice+1]) #calculate the number of patches to the left of the hinge
    index = 2*patches_left #get the index at which to start rotating
    #index = np.argmax(patch_arr[0,:]>0) #Get the index at which x becomes positive--everything to the right of this we rotate
    
    #rotate line and shape arrays using matrix multiplication. Matrix multiplication performed on all points to the right of the hinge. Indexing reflects position of hinge in arrays
    patch_arr[:,index:] = np.matmul(rotation_matrix,patch_arr[:,index:])
    shape_arr[:,np.sum(linelist[:hingechoice+1]*2):] = np.matmul(rotation_matrix,shape_arr[:,np.sum(linelist[:hingechoice+1]*2):])
    hinge_loc[:,hingechoice:] = np.matmul(rotation_matrix, hinge_loc[:,hingechoice:])

    return patch_arr, shape_arr, hinge_vec, hinge_loc


def translate_back(patch_arr_init,shape_arr_init,hinge_loc_init):
    '''This function translates any structure back to the reference position where the leftmost point of the first shape is located at the origin
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
    Outputs:
        new_line_arr: translated line_arr
        new_shape_arr: translated shape_arr
    '''

    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_loc = copy.deepcopy(hinge_loc_init)
    #Create a deepcopy of all three arrays (necessary to prevent modifying original when performing trial runs in Monte Carlo)
   
    xy_trans = np.array(([shape_arr[0,0]],[shape_arr[1,0]]))#Determine how far to translate each shape in order to place the first point of the first shape on the origin.
                                                          #This should be the negative of the location of the first point
   
    patch_arr -= xy_trans #translate line_arr by subtracting the value (subtracting a negative to make a positive)
    shape_arr -= xy_trans #translate shape_arr
    hinge_loc -= xy_trans
    
    return patch_arr,shape_arr, hinge_loc

def rotate_once(patch_arr,shape_arr,linelist,hinge_vec,hingechoice, hinge_loc, angle, patch_num):
    '''This function performs the translate,rotate,translate cycle all together and is important in later functions and for plotting or troubleshooting
    Inputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of shapes
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec: Vector of hinge angles. Interdipolar angle between each shape
        hingechoice: integer chosen hinge about which to revolve
        hinge_loc: 2x(m-1) array of x and y points that contain hinge locations. m is the number of shapes
        angle: float chosen angly by which to revolve
    #Outputs:
        new patch-arr: rotated line_arr
        new_shape_arr: rotated shape_arr
        new_hinge_vec: rotated hinge_vec
    '''

    patch_arr, shape_arr, hinge_loc = translate_to_origin(patch_arr, shape_arr, hinge_loc, hingechoice) #translate hinge to the origin
    patch_arr, shape_arr, hinge_vec, hinge_loc = rotate(patch_arr, shape_arr, linelist, hinge_vec, hinge_loc, hingechoice, angle, patch_num) #rotate structure
    patch_arr,shape_arr, hinge_loc = translate_back(patch_arr,shape_arr, hinge_loc) #translate structure back
    
    return patch_arr,shape_arr,hinge_vec,hinge_loc


def initialize_energy(magvec):
    '''This function initializes a series fo matrices that help to reduce computational expense of later code. This is necessary for the
        magentic energy calculation performed at each step of the MC simulation. This initialized energy already acounts for the magentization
        values of the patches
    #Inputs:
        magvec: a vector contianing the magnetic moments/length of each point at the end of a patch. This should be input manually or scaled based on SQUID data (A*m)
    Outputs:
        #mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        #v_xmat: An nx2 array of 1's
        #h_xmat: A 2xn array of -1's
        #v_ymat: An nx2 array of 1's
        #h_ymat: A 2xn array of -1's
        #Ml_mat: An nxn array that contains combinations of magnetizations and lengths
    '''

    n = len(magvec) #Determine the number of points (n)
    mask_arr = np.ones((n, n)) #Initialize mask_arr with ones

    M0 = (4*np.pi)*10**(-7) #J*m #magnetic constant
    constant = M0/(4*np.pi) #J*m #divide magentic constant by 4pi
    
    #Perform matrix multiplication to obtain multiplication between every possible point combination. Additionaly, multiply constant
    #Ml[:,None] essential to perform matrix multiplication because Ml is 1D
    Ml_mat = np.matmul(magvec[:,None],magvec[None,:])*constant #M1_mat is an array that describes the M/L combo for each pairwise interaction

    #Turn mask-arr into one that alternates between 1 and negative 1
    for i in range(n):
            for l in range(n):
                if (i+l) % 2 != 0:
                    mask_arr[i,l] *= -1

    #mask array must be a special upper diagnonal matrix. entries below the diagonal are ommitted because they're the negative of the upper. entries on the diagonal are
    #ommitted because they would be a point interacting with itself. entries one above the diagonal describe a point on one line interacting with a point on the same line and
    #must be ommitted
    k = 0 #Initialize counter
    for i in range(n): #columns
        #Every two rows, k increases by 2 such that the ommitted values are shifted down by two rows
        for j in range(k,n):
            mask_arr[j,i] = 0 #Replace unwanted entry with a zero
        if i>0 and i%2 != 0: #Increment k counter
            k+=2

    #Create horizontal and vertical matrices needed to subtract each x point from another using matrix math
    #Each vertical array will have its [:,0] entry as x or y-values. Every horizontal array will have its [1,:] entry as x or y values. Through matrix multiplication, this will
    #create an array with each entry as one point subtracted from another
    v_xmat = np.ones((n,2))
    h_xmat = np.ones((2,n))*-1
    v_ymat = np.ones((n,2))
    h_ymat = np.ones((2,n))*-1
            
    return mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat


def energy_math(patch_arr,mask_arr, v_xmat, h_xmat, v_ymat, h_ymat,Ml_mat):
    '''This function calculates the total interdipolar energy of the system
    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
    #Outputs:
        #E: total energy of the system
    '''
    #Create a deepcopy of line_arr to prevent line_arr from being modified during the vector operations in this function
    patch_arr = copy.deepcopy(patch_arr)
    
    v_xmat[:,0] = patch_arr[0,:] #replace first column with x-values
    h_xmat[1,:] = patch_arr[0,:] #replace second row with x-values
    xmat = np.matmul(v_xmat,h_xmat)*1e-6 #perform matrix multiplication to obtain each x-value subtracted from another. Multiply by 1e-6 to dimensionalize
    xmat_upper = np.multiply(xmat,mask_arr) #Multiply by the mask array to obtain the special upper diagnonal matrix and gain -1/1 pattern (this pattern is removed when squaring
                                            #and with have to be reinstated later)
    x_square = np.square(xmat_upper) #square all values. note that squaring removes -1/1 pattern
    
    v_ymat[:,0] = patch_arr[1,:] #replace first column with y-values
    h_ymat[1,:] = patch_arr[1,:] #replace second row with y-values
    ymat = np.dot(v_ymat,h_ymat)*1e-6 #perform matrix multiplication to obtain each y-value subtracted from another. Multiply by 1e-6 to dimensionalize
    ymat_upper = np.multiply(ymat,mask_arr) #Multiply by the mask array to obtain the special upper diagnonal matrix and gain -1/1 pattern (this pattern is removed when squaring
                                            #and with have to be reinstated later)
    y_square = np.square(ymat_upper ) #square all values. note that squaring removes -1/1 pattern
    
    final_mat = np.multiply(np.sqrt(x_square + y_square),mask_arr) #add the squares of x and y values and take the square root. Then multiply this element-wise by mask_arr
                                                                   #to reinstate -1/1 pattern
    indices = np.nonzero(final_mat) #Determine the indices of all nonzero values to allow for taking the reciprocal
    vec = np.multiply(Ml_mat[indices],np.reciprocal(final_mat[indices])) #multiply piecewise the nonzero values of the inverse of the magnitude matrix with corresponding values of the 
                                                                         #M/L combination matrix
  
    E = np.sum(vec) #Sum all values to determine the total interaction energy
    
    return E


def simulate_greedyDescent(patch_arr_init,shape_arr_init,linelist,hinge_vec_init, hinge_loc, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, tol=0):
    """
    Simulate as follows: for each hinge, sample a random angle, then move the hinge by that angle in the favorable direction and calculate the energy change.
    Accept the move with the largest negative energy change, then repeat.

    Inputs:

    Outputs:

    """

    # Deepcopy initial state
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_vec = copy.deepcopy(hinge_vec_init)

    # calculate the pre-rotation energy and store number of hinges
    current_energy = energy_math(patch_arr, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
    num_hinges = len(hinge_vec)
    polycount = count_shapes(shape_arr)

    for iteration in range(max_iter): # for our iterations

        # initialize variables to store the best moves
        best_deltaE = 0
        best_move = None

        for h in range(num_hinges): # for each hinge
        
            angle_trial = np.random.normal(0,std) #pull a trial angle from the Gaussian distribution

            for sign in [1,-1]: # test the angle in both directions
                angle_trial = angle_trial*sign 
                trial_patch, trial_shape, trial_hinge = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, angle_trial, patch_num) # rotate by the angle
                overlap = check_overlap(trial_shape, polycount)

                if overlap: #if the shapes overlap
                    steric_counter = 0
                    while steric_counter<10: # do ten attempts
                        new_angle_trial = angle_trial/2 # reduce the tested angle by half
                        trial_patch, trial_shape, trial_hinge = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, new_angle_trial, patch_num) #try rotating again
                        overlap = check_overlap(trial_shape, polycount)
                        if overlap: # if still overlapped
                            steric_counter += 1 # increment test counter and move to the next angle reduction
                        else: # if no overlap now with a smaller angle
                            break
                
                trial_energy = energy_math(trial_patch, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat) # calc new energy

                deltaE = trial_energy - current_energy # calculate the change in energy

                if deltaE < best_deltaE: # if it's a more favorable move
                    best_deltaE = deltaE # update best change in energy
                    best_move = (trial_patch, trial_shape, trial_hinge, trial_energy) # store the best move arrays

        if best_move is not None and best_deltaE < tol: # if there was a best move and the energy change was less than some defined minimum
            patch_arr, shape_arr, hinge_vec, current_energy = best_move # update our stored "current" values
        else: # no favorable moves
            print(f"Converged after {iteration} iterations.")
            break

        shapeplots(shape_arr, linelist, mag_vecs = patch_arr)

    return patch_arr, shape_arr, hinge_vec, current_energy