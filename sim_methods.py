import numpy as np
import matplotlib.pyplot as plt
import shapely as sp
from shapely.ops import polygonize
import copy
from visualization_functions import energy_math, count_shapes, rotate_once, check_overlap, shapeplots, update
from matplotlib.animation import FuncAnimation
import os
import shutil


def sim_many(sims, method, patch_arr_init,shape_arr_init,linelist,hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, kBT=4.11e-21, tol=0, plot=False, animate=False):
    """
    Simulate many times and store each simulation's hinge vector and final energy

    Inputs:
        sims: (int) number of simulations to run
        method: (str) kind of simulation to use; can be 'greedy descent', 'monte carlo', 'weighted sync'
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec_init: Vector of hinge angles. Interdipolar angle between each shape
        hinge_loc_init: 2x(m-1) array of x and y points that contain hinge locations. m is the number of shapes
        std: (float) standard deviation of the distribution from which to pull the rotation angle
        patch_num: n-dim list of patches per shape, where n is number of shapes
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        max_iter: (int) maximum number of iterations to run the simulation
        kBT: default room temperature
        tol: (float) defined minumum energy change value to be accepted (default zero)
        plot: (bool) whether or not to plot each sim
        animate: (bool) whether or not to save the animation of the final sim

    Outputs:
        final_hinges: An Nxn array of hinge angle values, where N is the number of simulations and n is the number of hinges
        final_e: An N-dim vector of final energies for each simulation, where N is the number of simulations
    """

    final_hinges = np.zeros((sims,len(hinge_vec_init))) #Initialize an array to store all final hinge conformations
    final_e = np.zeros(sims) #Initialize vector to store final energy state of each fold
    energies = np.empty(sims, dtype=object)
    
    if animate:
        # set the folder
        if method == 'greedy descent':
            # make a new empty folder for animation saves
            folder = 'GreedyDescentAnimations'
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        elif method == 'monte carlo':
            # make a new empty folder for animation saves
            folder = 'MonteCarloAnimations'
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        elif method == 'weighted sync':
            folder = 'WeightedSyncAnimations'
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
    else:
        folder = ''

    # run the simulation
    for i in range(sims): #For loop runs through all simulations
        if method == 'greedy descent':
            patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy, energy = simulate_greedyDescent(patch_arr_init,shape_arr_init,linelist,hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, tol=tol, animate=animate, sim_num=i, ani_folder=folder)
        elif method == 'monte carlo':
            patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy, energy = simulate_monteCarlo(patch_arr_init,shape_arr_init,linelist,hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, kBT=kBT, animate=animate, sim_num=i, ani_folder=folder)
        elif method == 'weighted sync':
            patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy, energy = simulate_weightedSync(patch_arr_init,shape_arr_init,linelist,hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, tol=tol, animate=animate, sim_num=i, ani_folder=folder)
        
        for j in range(len(hinge_vec)): #loop through number of movable hinges
            final_hinges[i,j]= hinge_vec[j] #Place all values of the final hinge angles into their corresponding index in final_hinges
            final_e[i] = current_energy #The minimum energy of a fold is stored
        if plot:
            shapeplots(shape_arr, linelist,hinge_loc, mag_vecs = patch_arr)

        energies[i] = energy
        
    return final_hinges, final_e, energies

def simulate_greedyDescent(patch_arr_init,shape_arr_init,linelist,hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, tol=0, animate=False, sim_num=0, ani_folder='GreedyDescentAnimations'):
    """
    Simulate as follows: for each hinge, sample a random angle, then move the hinge by that angle in the favorable direction and calculate the energy change.
    Accept the move with the largest negative energy change, then repeat.

    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec_init: Vector of hinge angles. Interdipolar angle between each shape
        hinge_loc_init: 2x(m-1) array of x and y points that contain hinge locations. m is the number of shapes
        std: (float) standard deviation of the distribution from which to pull the rotation angle
        patch_num: n-dim list of patches per shape, where n is number of shapes
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        max_iter: (int) maximum number of iterations to run the simulation
        tol: (float) defined minumum energy change value to be accepted (default zero)
        animate: (bool) whether or not to animate the folding
        sim_num: (int) for saving animations by simulation number
        ani_folder: (str) location to save animations

    Outputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the final magentic patches
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each final shape
        hinge_vec: Vector of final hinge angles
        hinge_loc: 2x(m-1) array of x and y points that contain final hinge locations
        current_energy: (float) Energy at the end of the simulation
    """

    # Deepcopy initial state
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_vec = copy.deepcopy(hinge_vec_init)
    hinge_loc = copy.deepcopy(hinge_loc_init)

    # calculate the pre-rotation energy and store number of hinges
    current_energy = energy_math(patch_arr, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
    num_hinges = len(hinge_vec)
    polycount = count_shapes(shape_arr)
    states=[]
    states.append((copy.deepcopy(shape_arr),copy.deepcopy(hinge_loc), copy.deepcopy(patch_arr))) # save initial state for animation
    energy = [current_energy]

    for iteration in range(max_iter): # for our iterations

        # initialize variables to store the best moves
        best_deltaE = 0
        best_move = None

        for h in range(num_hinges): # for each hinge
        
            trial_angle = np.random.normal(0,std) #pull a trial angle from the Gaussian distribution

            for sign in [1,-1]: # test the angle in both directions
                test_trial_angle = trial_angle*sign 
                trial_patch, trial_shape, trial_hinge, trial_hingeloc = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, test_trial_angle, patch_num) # rotate by the angle
                overlap = check_overlap(trial_shape, polycount)

                if overlap: #if the shapes overlap
                    steric_counter = 0
                    new_trial_angle = test_trial_angle
                    while steric_counter<10: # do ten attempts
                        new_trial_angle = new_trial_angle/2 # reduce the tested angle by half
                        trial_patch, trial_shape, trial_hinge, trial_hingeloc = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, new_trial_angle, patch_num) #try rotating again
                        overlap = check_overlap(trial_shape, polycount)
                        if overlap: # if still overlapped
                            steric_counter += 1 # increment test counter and move to the next angle reduction
                        else: # if no overlap now with a smaller angle
                            break
                    if steric_counter == 10:
                        continue
                
                trial_energy = energy_math(trial_patch, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat) # calc new energy

                deltaE = trial_energy - current_energy # calculate the change in energy

                if deltaE < best_deltaE: # if it's a more favorable move
                    best_deltaE = deltaE # update best change in energy
                    best_move = (trial_patch, trial_shape, trial_hinge, trial_hingeloc, trial_energy) # store the best move arrays

        if best_move is not None and best_deltaE < tol: # if there was a best move and the energy change was less than some defined minimum
            patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy = best_move # update our stored "current" values
            energy.append(current_energy)
        else: # no favorable moves
            #print(f"Converged after {iteration} iterations.")
            break
        
        # check if we are on the order of kBT or not
        # if abs(best_deltaE) < 10e-20:
        #     print('small E: ',best_deltaE, iteration)

        states.append((copy.deepcopy(shape_arr),copy.deepcopy(hinge_loc), copy.deepcopy(patch_arr))) # save state for animation

    # animate the folding if asked for
    if animate:
        fig, ax = plt.subplots(figsize=(10,10))
        bounds = [None]
        ani = FuncAnimation(fig, update, frames=iteration+1, interval=100, fargs=(states, linelist, ax, bounds))
        ani.save(f"{ani_folder}/greedyDescent_folding_{sim_num}.gif", writer="pillow", fps=10)

    #print("Current energy: ", current_energy, " Joules")
        
        # plot line for testing commented out typically
        ### shapeplots(shape_arr, linelist, mag_vecs = patch_arr)

    return patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy, np.array(energy)


def simulate_monteCarlo(patch_arr_init, shape_arr_init, linelist, hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, kBT=4.11e-21, animate=False, sim_num=0, ani_folder='MonteCarloAnimations'):
    """
    Simulate as follows: randomly pick a hinge, propose a random rotation, and accept/reject based on the Boltzmann distribution.

    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec_init: Vector of hinge angles. Interdipolar angle between each shape
        hinge_loc_init: 2x(m-1) array of x and y points that contain hinge locations. m is the number of shapes
        std: (float) standard deviation of the distribution from which to pull the rotation angle
        patch_num: n-dim list of patches per shape, where n is number of shapes
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        max_iter: (int) maximum number of iterations to run the simulation
        kBT: (float) kBT constant at whatever temperature (default room temperature)
        animate: (bool) whether or not to store each frame for animation
        sim_num: (int) for saving animations by simulation number
        ani_folder: (str) location to save animations

    Outputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the final magentic patches
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each final shape
        hinge_vec: Vector of final hinge angles
        hinge_loc: 2x(m-1) array of x and y points that contain final hinge locations
        current_energy: (float) Energy at the end of the simulation
    """
    # Deepcopy initial state
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_vec = copy.deepcopy(hinge_vec_init)
    hinge_loc = copy.deepcopy(hinge_loc_init)

    # Store the current energy of the conformation
    current_energy = energy_math(patch_arr, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
    energy = [current_energy]

    # store hinge number and shape number
    num_hinges = len(hinge_vec)
    polycount = count_shapes(shape_arr)
    states = []
    states.append((copy.deepcopy(shape_arr),copy.deepcopy(hinge_loc), copy.deepcopy(patch_arr))) # save initial state for animation

    accepted = 0 # initialized acceptance count

    for iteration in range(max_iter): # for each interation

        # Pick random hinge
        h = np.random.randint(0, num_hinges)

        # Propose random move
        trial_angle = np.random.normal(0, std)

        # Rotate the hinge
        trial_patch, trial_shape, trial_hinge, trial_hingeloc = rotate_once(patch_arr, shape_arr,linelist, hinge_vec, h, hinge_loc, trial_angle, patch_num)

        # Steric check 
        overlap = check_overlap(trial_shape, polycount)

        if overlap: #if the shapes overlap
            steric_counter = 0
            new_trial_angle = trial_angle
            while steric_counter<10: # do ten attempts
                new_trial_angle = new_trial_angle/2 # reduce the tested angle by half
                trial_patch, trial_shape, trial_hinge, trial_hingeloc = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, new_trial_angle, patch_num) #try rotating again
                overlap = check_overlap(trial_shape, polycount)
                if overlap: # if still overlapped
                    steric_counter += 1 # increment test counter and move to the next angle reduction
                else: # if no overlap now with a smaller angle
                    break
            if steric_counter == 10:
                continue

        # Compute energy and energy change
        trial_energy = energy_math(trial_patch, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
        deltaE = trial_energy - current_energy

        # Metropolis acceptance
        if deltaE <= 0:
            accept = True
        else:
            prob = np.exp(-deltaE / kBT)
            accept = np.random.rand() < prob

        # Update state if accepted
        if accept:
            patch_arr = trial_patch
            shape_arr = trial_shape
            hinge_vec = trial_hinge
            hinge_loc = trial_hingeloc
            current_energy = trial_energy
            accepted += 1
            energy.append(current_energy)
        else:
            continue

        # # Additional diagnostics
        # if iteration % 1000 == 0 and iteration > 0:
        #     acc_rate = accepted / iteration
        #     print(f"Iter {iteration} | E = {current_energy:.3e} | acc = {acc_rate:.3f}")

        states.append((copy.deepcopy(shape_arr),copy.deepcopy(hinge_loc), copy.deepcopy(patch_arr))) # save state for animation

    # animate the folding if asked for
    if animate:
        fig, ax = plt.subplots(figsize=(10,10))
        bounds = [None]
        ani = FuncAnimation(fig, update, frames=accepted+1, interval=100, fargs=(states, linelist, ax, bounds))
        ani.save(f"{ani_folder}/monteCarlo_folding_{sim_num}.gif", writer="pillow", fps=10)

    #print("Current energy:", current_energy, "Joules")
    #print(f"Final acceptance rate: {accepted/max_iter:.3f}")

    return patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy, np.array(energy)


def simulate_weightedSync(patch_arr_init, shape_arr_init, linelist, hinge_vec_init, hinge_loc_init, std, patch_num, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat, max_iter, tol=0, animate=False, sim_num=0, ani_folder='WeightedSyncAnimations'):
    """
    Simulate as follows: sample a random angle, then test each hinge rotation by that angle in both directions, and store the most favorable energy change.
    Then, rotate all hinges simultaneously by a weighted amount (weighted by most favorable energy change). If overlap, reduce all hinge rotations by half and try again.

    Inputs:
        patch_arr_init: 2x(2n) array of x and y points that describe the lines of the magentic patch. n is number of distinct magnetic domains
        shape_arr_init: 2x(2n*s) array of x and y points that describe the lines enclosing each shape. n is number of shapes. s is number of sides on that shape
        linelist: List of integers describing the number of sides in each shape
        hinge_vec_init: Vector of hinge angles. Interdipolar angle between each shape
        hinge_loc_init: 2x(m-1) array of x and y points that contain hinge locations. m is the number of shapes
        std: (float) standard deviation of the distribution from which to pull the rotation angle
        patch_num: n-dim list of patches per shape, where n is number of shapes
        mask_arr: An special nxn upper diagonal matrix of 0's, 1's, and -1's useful for the energy calculation. n is 2x number of shapes
        v_xmat: An nx2 array of 1's
        h_xmat: A 2xn array of -1's
        v_ymat: An nx2 array of 1's
        h_ymat: A 2xn array of -1's
        Ml_mat: An nxn array that contains combinations of magnetizations and lengths
        max_iter: (int) maximum number of iterations to run the simulation
        tol: (float) defined minumum energy change value to be accepted (default zero)
        animate: (bool) whether or not to store each frame for animation
        sim_num: (int) for saving animations by simulation number
        ani_folder: (str) location to save animations

    Outputs:
        patch_arr: 2x(2n) array of x and y points that describe the lines of the final magentic patches
        shape_arr: 2x(2n*s) array of x and y points that describe the lines enclosing each final shape
        hinge_vec: Vector of final hinge angles
        hinge_loc: 2x(m-1) array of x and y points that contain final hinge locations
        current_energy: (float) Energy at the end of the simulation
    """
    
    # Deep copy initial state
    patch_arr = copy.deepcopy(patch_arr_init)
    shape_arr = copy.deepcopy(shape_arr_init)
    hinge_vec = copy.deepcopy(hinge_vec_init)
    hinge_loc = copy.deepcopy(hinge_loc_init)

    # calculate the current energy
    current_energy = energy_math(patch_arr, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
    energy = [current_energy]

    # store the numer of hinges and shapes
    num_hinges = len(hinge_vec)
    polycount = count_shapes(shape_arr)
    states = []
    states.append((copy.deepcopy(shape_arr),copy.deepcopy(hinge_loc), copy.deepcopy(patch_arr))) # save initial state for animation

    for iteration in range(max_iter):
        
        trial_angle = np.random.normal(0, std) # pull a random angle

        # initialize energy and sign arrays
        deltaEs = np.zeros(num_hinges)
        signs = np.zeros(num_hinges)

        # initialize scale factor array for when steric hindrance is an issue
        scale_factor = np.ones(num_hinges)

        # evaluate each hinge independently
        for h in range(num_hinges):

            # initialize "best" values for the hinge
            best_deltaE = np.inf
            best_sign = 0

            for sign in [1, -1]: # for both directions
                test_trial_angle = trial_angle * sign

                # rotate the shape
                trial_patch, trial_shape, _, _ = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, test_trial_angle, patch_num)

                overlap = check_overlap(trial_shape, polycount) # check for overlap

                if overlap: #if the shapes overlap
                    steric_counter = 0
                    new_trial_angle = trial_angle
                    while steric_counter<10: # do ten attempts
                        new_trial_angle = new_trial_angle/2 # reduce the tested angle by half
                        scale_factor[h] = scale_factor[h]/2 # save the scaling factor for later
                        trial_patch, trial_shape, trial_hinge, trial_hingeloc = rotate_once(patch_arr, shape_arr, linelist, hinge_vec, h, hinge_loc, new_trial_angle, patch_num) #try rotating again
                        overlap = check_overlap(trial_shape, polycount)
                        if overlap: # if still overlapped
                            steric_counter += 1 # increment test counter and move to the next angle reduction
                        else: # if no overlap now with a smaller angle
                            break
                    if steric_counter == 10:
                        continue

                # calculate the energy change
                trial_energy = energy_math(trial_patch, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
                deltaE = trial_energy - current_energy

                if deltaE < best_deltaE: # if it's a more favorable move, update the "best" move
                    best_deltaE = deltaE
                    best_sign = sign

            if best_deltaE < 0: # as long as the best move is favorable
                deltaEs[h] = -best_deltaE   # positive weight
                signs[h] = best_sign
            else:
                deltaEs[h] = 0
                signs[h] = 0

        # normalize weights
        total = np.sum(deltaEs)

        if total == 0:
            break  # no favorable directions anywhere

        weights = deltaEs / np.sum(deltaEs) #max(deltaEs) # normalization

        # construct simultaneous rotation
        theta_vec = trial_angle * weights * signs * scale_factor # angle array for rotation

        # parameters for keeping track of overlap failure
        success = False
        scale = 1.0

        for _ in range(10): # try ten times

            trial_patch = copy.deepcopy(patch_arr)
            trial_shape = copy.deepcopy(shape_arr)
            trial_hinge = copy.deepcopy(hinge_vec)
            trial_hingeloc = copy.deepcopy(hinge_loc)

            # apply all hinge rotations sequentially
            for h in range(num_hinges):
                if theta_vec[h] == 0: # if the hinge isn't rotating, continue
                    continue

                # attempt rotation
                trial_patch, trial_shape, trial_hinge, trial_hingeloc = rotate_once(trial_patch, trial_shape, linelist, trial_hinge, h, trial_hingeloc, theta_vec[h] * scale, patch_num)

            overlap = check_overlap(trial_shape, polycount) # check overlap

            if not overlap:
                success = True
                break
            else:
                scale *= 0.5  # shrink all moves and try again if overlap

        if not success:
            continue  # reject step

        # calculate energy change
        trial_energy = energy_math(trial_patch, mask_arr, v_xmat, h_xmat, v_ymat, h_ymat, Ml_mat)
        deltaE_total = trial_energy - current_energy

        if deltaE_total < tol: # accept move if the energy change is favorable
            patch_arr = trial_patch
            shape_arr = trial_shape
            hinge_vec = trial_hinge
            hinge_loc = trial_hingeloc
            current_energy = trial_energy
            energy.append(current_energy)
        else:
            break # if no favorable energy change, stop

        states.append((copy.deepcopy(shape_arr),copy.deepcopy(hinge_loc), copy.deepcopy(patch_arr))) # save state for animation

    #print("Current energy:", current_energy, "Joules")

    # animate the folding if asked for
    if animate:
        fig, ax = plt.subplots(figsize=(10,10))
        bounds = [None]
        ani = FuncAnimation(fig, update, frames=iteration+1, interval=100, fargs=(states, linelist, ax, bounds))
        ani.save(f"{ani_folder}/weightedSync_folding_{sim_num}.gif", writer="pillow", fps=10)

    return patch_arr, shape_arr, hinge_vec, hinge_loc, current_energy, np.array(energy)