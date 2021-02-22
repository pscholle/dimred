from lasso.dyna.D3plot import D3plot
# from lasso.dyna.ArrayTypeFilter import ArrayTypeFilter
from lasso.dyna.ArrayType import ArrayType
from typing import List, Dict, Union, Tuple, Set
import math
import numpy as np

from scipy.spatial import ConvexHull
from scipy.stats import binned_statistic_2d, BinnedStatistic2dResult

from sphere import create_sphere, to_spherical_coordinates


def load_arrays_from_plot(plt: Union[D3plot, None] = None,
                          filepath: str = "",
                          state_array_filter: List[str] = [],
                          copy: bool = True) -> Dict[str, np.ndarray]:
    '''Extract arrays from a d3plot.

    Parameters
    ----------
    plt:Union[D3plot,None]
        D3plot
    filepath:str
        File path of d3plot.
    array_filter:List[str]
        List containing the array names to be extracted from the d3plot arrays.
    copy:bool
        Copy the arrays into a new array.
    '''
    plt = D3plot(filepath) if filepath and plt == None else plt

    if plt == None:
        return {}

    data = {}
    if state_array_filter:
        for arr_name in state_array_filter:
            if arr_name in plt.arrays:
                data[arr_name] = np.array(plt.arrays[arr_name], copy=copy)

        # make sure geometry data is loaded like element part and element node indexes
        for arr_name in ArrayTypeFilter.element_part_indexes:
            if arr_name not in plt.arrays:
                data[arr_name] = np.array(plt.arrays[arr_name], copy=copy)

        for arr_name in ArrayTypeFilter.element_node_indexes:
            if arr_name not in plt.arrays:
                data[arr_name] = np.array(plt.arrays[arr_name], copy=copy)
    else:
        # no filter so load the get all the arrays
        for arr_name, arr in plt.arrays.items():
            data[arr_name] = np.array(arr, copy=copy)

    return data


# def extract_part_arrays(d3plot_arrays: Dict[str, np.ndarray], part_index_filter: Set[int] = set()) -> Dict[int, Dict[str, np.ndarray]]:
#     '''
#     '''
#     part_ids = [0]

#     if ArrayType.part_ids in d3plot_arrays:
#         part_ids = d3plot_arrays[ArrayType.part_ids] if part_index_filter == set(
#         ) else part_index_filter

#     master: Dict[int, Dict[str, np.ndarray]] = {pid: {} for pid in part_ids}

#     for part_index, part_id in enumerate(part_ids):
#         for ii, (element_part_indexes, node_indexes) in enumerate(zip(ArrayTypeFilter.element_part_indexes, ArrayTypeFilter.element_node_indexes)):
#             element_indexes = np.where(
#                 d3plot_arrays[element_part_indexes] == part_index)[0]
#             if len(element_indexes) != 0:
#                 # node connectivity of an element
#                 connectivity = d3plot_arrays[node_indexes][element_indexes]
#                 for name, arr in d3plot_arrays.items():
#                     d = arr[:, connectivity]
#                     master[part_id][name] = d

#     return master


def build_base(plt_arrays: Dict[str, np.ndarray]) -> np.ndarray:
    '''Computes the centroids of all the elements of a model.

    Note
    ----
    This methods is meant to handle parts that only have shells.
    '''

    part_ids = plt_arrays[ArrayType.part_ids] if ArrayType.part_ids in plt_arrays else [
        0]

    n_elements: int = 3

    # where we will store the centroids
    element_centroid = np.empty(
        (n_elements, 3), dtype=np.float32)

    # we will only use the shells for projecting fields
    for part_index, _ in enumerate(part_ids):
        # element_part_indexes stores the indexes of
        # a part id not the id itself
        part_shell_indexes = np.where(
            plt_arrays[ArrayType.element_shell_part_indexes] == part_index)[0]

        if len(part_shell_indexes) == 0:
            continue

        # node indexes are stored in the element shell node indexes array
        # we have the element indexes and can use that to get our node indexes
        part_node_indexes = plt_arrays[ArrayType.element_shell_node_indexes][part_shell_indexes]

        # indexes of all shells which are triangles
        # dyna will store the index of the third node twice
        # i.e. [2231,1221,222,222]
        tria_shell_indexes = np.where(
            part_node_indexes[:, 2] - part_node_indexes[:, 3] == 0)[0]

        # indexes of all nodes that are part of triangle shell elements
        node_indexes_tria = part_node_indexes[tria_shell_indexes]

        # get the coordinates of the nodes
        tria_node_coordinates = plt_arrays[ArrayType.node_coordinates][node_indexes_tria]

        # compute the mean of all tria node coordinates ignoring index 3
        # as this is the same as index 2
        tria_shell_centroid_coordinates = np.mean(
            tria_node_coordinates[:, :2], axis=1)

        # find all shells where index 2 and 3 are not equal
        quad_shell_indexes = np.where(
            part_node_indexes[:, 2] - part_node_indexes[:, 3] != 0)[0]

        quad_node_indexes = part_node_indexes[quad_shell_indexes]
        quad_node_coordinates = plt_arrays[ArrayType.node_coordinates][quad_node_indexes]

        quad_shell_centroid_coordinates = np.mean(
            quad_node_coordinates, axis=1)

        # populate the master array
        element_centroid[tria_shell_indexes] = tria_shell_centroid_coordinates
        element_centroid[quad_shell_indexes] = quad_shell_centroid_coordinates

    return element_centroid


def create_historgram(cloud: np.ndarray, sphere_axis: str = 'Z') -> BinnedStatistic2dResult:
    '''
    '''
    centroid = np.mean(
        cloud, axis=0)

    hull = ConvexHull(cloud)

    # we need to determine the largest distance in this point
    # cloud so we can give the sphere a dimension
    # we can also create a sphere of random size but this could
    # scew the results
    dist = np.linalg.norm(hull.max_bound - hull.min_bound)

    bins_a, bins_b = create_sphere(dist)

    cloud_alpha, cloud_beta = to_spherical_coordinates(
        cloud, centroid, AXIS=sphere_axis)

    return binned_statistic_2d(cloud_alpha, cloud_beta, None, 'count', bins=[
        bins_a, bins_b], expand_binnumbers=True)


def main():
    path = 'plots/d3plot'

    d = load_arrays_from_plot(filepath=path)

    part_indexes = np.array([0, 0, 0])

    node_indexes = np.array(
        [[0, 1, 2, 3], [2, 1, 3, 3], [1, 12, 11, 3]], dtype=np.int)

    node_coordinates = np.random.randn(13, 3)

    centroids = build_base({"element_shell_node_indexes": node_indexes,
                            "element_shell_part_indexes": part_indexes,
                            "node_coordinates": node_coordinates})

    return


if __name__ == '__main__':
    main()
