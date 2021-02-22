import numpy as np
from sklearn.preprocessing import normalize
from typing import List, Dict, Any
import os
from lasso.dyna.D3plot import D3plot
from lasso.dyna.ArrayType import ArrayType as atype

import glob
import sys
import time
import h5py
from scipy.spatial import ConvexHull
from scipy.stats import binned_statistic_2d
from sklearn.preprocessing import normalize
import psutil


def create_sphere(diameter: float):
    """Creates two vectors along the alpha and beta axis of a sphere. Alpha represents 
     the angle from the sphere axis to the equator. Beta between vectors from the center of the sphere to one
     of the poles and the equator.

     Parameters 
     ----------
     diameter:
        Diameter of the sphere.

    Returns 
    -------
    bin_beta: np.ndarray    
        Bin bounds for the beta angles.

    bin_alpha: np.ndarray
        Bin bounds for the alpha angles.

    """
    # number of partitions for equator
    n_alpha = 145
    # number of partitions for longitude
    n_beta = 144

    r = diameter/2.0

    # area of sphere
    a_sphere = 4 * np.pi * r**2
    n_ele = n_beta**2
    a_ele = a_sphere/n_ele

    # alpha angles around the equator and the size of one step
    bin_alpha, delt_alpha = np.linspace(0, 2*np.pi, n_alpha, retstep=True)

    # bins for beta axis in terms of axis coorindates between -1 and 1
    count = np.linspace(0.0, float(n_beta), 145)
    tmp = count * a_ele
    tmp /= r**2 * delt_alpha
    bin_beta = 1 - tmp
    if bin_beta[-1] < -1:
        bin_beta[-1] = -1

    bin_beta = np.arccos(bin_beta)
    return bin_alpha, bin_beta


def sphere_hashing(bin_numbers: np.ndarray, bin_counts: np.ndarray, field: np.ndarray):
    """Porduce the binned val

    Parameters
    ----------
    bin_numbers:
        This array holds the indexes of the bins for every point of a model
    bin_counts:
        Dont know what this is anymore

    """
    assert len(bin_numbers[0] == len(field))

    n_rows = bin_counts.shape[0]
    n_cols = bin_counts.shape[1]

    # bin x and y indexes for each point in field
    binx = np.asarray(bin_numbers[0]) - 1
    biny = np.asarray(bin_numbers[1]) - 1

    # bincout for averaging
    bin_count = np.zeros((n_rows, n_cols))

    # averaged result to return
    binned_field = np.zeros((n_rows, n_cols))
    binned_field[binx[:], biny[:]] += field[:]
    bin_count[binx[:], biny[:]] += 1

    binned_field = binned_field.flatten()
    bin_count = bin_count.flatten()

    # exclude all zero entries
    nonzero_inds = np.where(bin_count != 0)

    # average the fields
    binned_field[nonzero_inds] /= bin_count[nonzero_inds]

    return binned_field


def to_spherical_coordinates(points: np.ndarray, centroid: np.ndarray, AXIS: str = 'Z'):
    """
    """
    indexes = [0, 1, 2]
    # indexes for correct projection based on user input
    if AXIS == 'Z':
        indexes = [0, 1, 2]  # z axis aligned with global z axis
    elif AXIS == 'Y':
        indexes = [0, 2, 1]  # z axis aligned with global y axis
    elif AXIS == 'X':
        indexes = [2, 1, 0]  # z axis aligned with global x axis

    n_points = len(points)
    cloud_alpha = np.empty(n_points)
    cloud_beta = np.empty(n_points)

    # vector from centroid to point
    vec = points - centroid

    # normalize the vector
    vec = normalize(vec, axis=1, norm='l2')

    # compute the angle on the local xy plane
    ang = np.arctan2(vec[:, indexes[1]], vec[:, indexes[0]])
    neg_indexes = np.where(ang < 0)
    ang[neg_indexes] += 2*np.pi

    # return values
    cloud_alpha = ang
    cloud_beta = np.arccos(vec[:, indexes[2]]/1)

    return cloud_alpha, cloud_beta


def memory_use():
    process = psutil.Process(os.getpid())
    process_memory = float(process.memory_info().rss) * 1e-6
    return process_memory


def compute_hashes(
        load_dir: str,
        save_dir: str,
        field_keys: list,
        run_ids: list,
        part_ids_collection: np.ndarray,
        collection_names: str,
        verbose: bool,
        use_femzip: bool):

    # get all file names in the save directory
    created_files = glob.glob(save_dir+'*.h5')

    os.chdir(save_dir)

    ts_main = time.time()

    for run in run_ids:
        filepath = load_dir + run

        # for runtime of data loading
        ts_l = time.time()

        # d3plot data
        plot = D3plot(filepath, use_femzip)

        # end of data loading
        te_l = time.time()

        n_timesteps = len(
            plot.arrays[atype.element_shell_effective_plastic_strain][0, 0, :])

        for part_ids, collection_name in zip(part_ids_collection, collection_names):

            new_file_name = run + '_hashed_' + collection_name + '.h5'

            # if already hashed, do the next part
            if new_file_name in created_files:
                continue

            hf = h5py.File(save_dir + run + '_hashed_' +
                           collection_name + '.h5', 'w')

            # do the thing
            master_coords = np.array([], dtype=np.float32).reshape(0, 3)

            master_displacements = np.empty(
                (0, 3, n_timesteps), dtype=np.float32)

            master_shell_indexes = np.array([], dtype=np.int).reshape(-1,)

            # concatenate all elements and fields into one array
            for part_id in part_ids:
                part_shell_indexes = np.where(
                    plot.arrays[atype.element_shell_part_indexes] - 1 == part_id)[0]

                n_elements = len(part_shell_indexes)
                if n_elements == 0:
                    continue

                part_node_indexes = (
                    plot.arrays[atype.element_shell_node_indexes] - 1)[part_shell_indexes]

                part_shell_coords = np.empty(
                    (n_elements, 3), dtype=np.float32)

                part_shell_displacements = np.empty(
                    (n_elements, 3, n_timesteps), dtype=np.float32)

                tria_local_indexes = np.where(
                    part_node_indexes[:, 2] - part_node_indexes[:, 3] == 0)[0]

                tria_node_displacement = plot.arrays[atype.node_displacement][part_node_indexes[tria_local_indexes]]

                tria_shell_centroid_displacements = np.mean(
                    tria_node_displacement[:, :2], axis=1)

                rect_local_indexes = np.where(
                    part_node_indexes[:, 2] - part_node_indexes[:, 3] != 0)[0]

                rect_node_displacement = plot.arrays[atype.node_displacement][part_node_indexes[rect_local_indexes]]

                rect_shell_centroid_displacements = np.mean(
                    rect_node_displacement, axis=1)

                tria_node_coords = plot.arrays[atype.node_coordinates][part_node_indexes[tria_local_indexes]]
                rect_node_coords = plot.arrays[atype.node_coordinates][part_node_indexes[rect_local_indexes]]

                rect_shell_centroid_coords = np.mean(rect_node_coords, axis=1)
                tria_shell_centroid_coords = np.mean(
                    tria_node_coords[:, :2], axis=1)

                if len(tria_local_indexes != 0):
                    part_shell_displacements[tria_local_indexes] = tria_shell_centroid_displacements
                    part_shell_coords[tria_local_indexes] = tria_shell_centroid_coords

                part_shell_displacements[rect_local_indexes] = rect_shell_centroid_displacements
                part_shell_coords[rect_local_indexes] = rect_shell_centroid_coords

                if not np.array_equal(part_shell_coords[0, :], part_shell_displacements[0, :, 0]):
                    print("There is a discrepency for part with id '{}'. The displacement at time step 0 does not equal the node coordinates. Skipping this part.".format(part_id))
                    continue

                # concatenate the indexes and displacements of this part to master
                master_coords = np.concatenate(
                    (master_coords, part_shell_coords), axis=0)

                master_displacements = np.concatenate(
                    (master_displacements, part_shell_displacements), axis=0)

                master_shell_indexes = np.concatenate(
                    (master_shell_indexes, part_shell_indexes), axis=0)

            hf.create_dataset("shell_coordinates", data=master_coords)
            hf.create_dataset("shell_displacements",
                              data=master_displacements)

            # convex hull can raise an error if the point cloud is not 3D
            # try:
            # sphere grid only needs to be created once
            centroid = np.mean(
                master_coords, axis=0)

            hull = ConvexHull(master_coords)
            dist = np.linalg.norm(hull.max_bound - hull.min_bound)

            bins_a, bins_b = create_sphere(dist)

            cloud_alpha, cloud_beta = to_spherical_coordinates(
                master_coords, centroid, AXIS='Z')

            histo = binned_statistic_2d(cloud_alpha, cloud_beta, None, 'count', bins=[
                bins_a, bins_b], expand_binnumbers=True)

            # loop over all provided keys and save the hashes to hdf5
            for k in field_keys:
                # measure time for hashing
                ts_h = time.time()
                hashes = None

                if k == "node_displacement":
                    hashes = np.empty((n_timesteps, 144*144))
                    tmp = master_displacements - \
                        master_coords[:, :, np.newaxis]

                    part_fields = np.linalg.norm(tmp, axis=1)

                    # do the thing
                    for step in range(n_timesteps):
                        hashes[step] = sphere_hashing(
                            histo.binnumber, histo.statistic, part_fields[:, step])

                elif k == "node_displacement_dims":
                    hashes = np.empty((n_timesteps, 3, 144*144))

                    tmp = np.abs(master_displacements -
                                 master_coords[:, :, np.newaxis])

                    for comp in range(3):
                        # hash each dimension
                        for step in range(n_timesteps):
                            hashes[step, comp] = sphere_hashing(
                                histo.binnumber, histo.statistic, tmp[:, comp, step])

                elif k == "element_shell_effective_plastic_strain":
                    hashes = np.empty((n_timesteps, 144*144))
                    part_fields = np.mean(
                        plot.arrays[atype.element_shell_effective_plastic_strain], axis=1)[master_shell_indexes]

                    # do the thing
                    for step in range(n_timesteps):
                        hashes[step] = sphere_hashing(
                            histo.binnumber, histo.statistic, part_fields[:, step])

                    hf.create_dataset(k, data=part_fields)

                hf.create_dataset("hashes_" + k, data=hashes)

                te_h = time.time()

                if verbose:
                    print("Hashed field '{}' for collection '{}'in {:.4f} seconds.".format(
                        k, collection_name, te_h - ts_h))

            del master_displacements, master_coords, master_shell_indexes

            # except Exception as e:
            #     print(str_error(str(e)))

            hf.close()
            print("")

        print("Processing of {} complete.".format(run))
        del plot

    te_main = time.time()

    print("Runtime [program] {0:.2f}".format(te_main - ts_main))

    return
