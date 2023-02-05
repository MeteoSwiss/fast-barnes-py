# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
A hyper-fast kd-tree implementation tailored for its application in 'radius' Barnes
interpolation algorithm and thus providing only radius-search functionality.
Uses Numba JIT compiler and numpy arrays to achieve the best performance with the
downside that Python class formalism cannot be used and has to be emulated using
data containers and 'external' functions acting on them.

Assuming the sample points are given by the array `pts`, repeated radius searches on
the corresponding kd-tree can be conducted in this way:

    
# create kd-tree 'instance'
kd_tree = create_kdtree(pts)

# create kd-tree search 'object'
radius = 12.5
kd_tree_search = prepare_search(radius, *kd_tree)

# extract array indices and their distances from returned tuple
res_index, res_sqr_dist, _, _, _ = kd_tree_search

# perform searches using search 'object'
search_pts = np.asarray([[3.2, 7.1], [9.8, -1.6], ...])
for pt in serach_pts:
    radius_search(pt, *kd_tree_search)
    
    # handle results
    for k in range(res_index[-1]):
        # do something with `res_index[k]` and `res_sqr_dist[k]`
        ...
    

Created on Sat May 28 15:35:23 2022
@author: Bruno Zürcher
"""

import numpy as np
from numba import njit


###############################################################################

@njit
def create_kdtree(pts):
    """
    Creates a balanced kd-tree 'instance' given the N point coordinates `pts`.
    
    In absence of a class concept supported by Numba, the kd-tree 'instance' merely
    consists of a tuple containing the tree structure and the points, which were
    passed to construct it.
    
    The tree is described by N nodes, which themselves are simple integer arrays of
    length 3. Array element 0 contains the index of the root node of the left
    sub-tree, element 1 the index of the root node of the right sub-tree and
    element 2 the index of the parent node. No node is represented by -1.
    The index of the root node of the whole kd-tree is stored in the additionally
    appended N+1-th array element.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x 2 containing the x- and y-coordinates
        (or if you like the longitude/latitude) of the N sample points.

    Returns
    -------
    tree : numpy ndarray
        A (N+1) x 3 integer array contiaining the index description of the kd-tree.
    pts : numpy ndarray
        The original point array that was used to construct the kd-tree.
    """

    num = len(pts)
    index_map = np.arange(num+1)
    tree = np.full((num+1,3), -1, dtype=np.int32)

    root = _median_sort(pts, index_map, 0, 0, num, tree)

    # reorder tree using reverse mapping
    tree[index_map] = tree.copy()
    # map tree entries using forward map; but first map -1 to -1
    index_map[-1] = -1
    for k in range(num):
        for i in range(3):
            tree[k,i] = index_map[tree[k,i]]

    # map also root index
    root = index_map[root]
    # and store root index in tree
    tree[-1,0] = root

    return tree, pts


@njit
def _median_sort(pts, index_map, cInd, frm, to, tree):
    """
    Determines median node by using "select median" algorithm, which establishes
    only partial sort.
    """
    if to-frm == 1:
        return frm
    
    # first sort specified array range with respect to coordinate index
    _partial_sort(pts, index_map, cInd, frm, to)
    median_ind = (frm + to) // 2
    median_pivot = pts[index_map[median_ind], cInd]
    # find 'left-most' node with same median value
    while median_ind > frm and pts[index_map[median_ind-1], cInd] == median_pivot:
        median_ind -= 1

    # recursively call median sort of left and right part
    if frm != median_ind:
        left_median = _median_sort(pts, index_map, 1-cInd, frm, median_ind, tree)
        tree[median_ind,0] = left_median
        tree[left_median,2] = median_ind

    if median_ind+1 != to:
        right_median = _median_sort(pts, index_map, 1-cInd, median_ind+1, to, tree)
        tree[median_ind,1] = right_median
        tree[right_median,2] = median_ind

    return median_ind


@njit
def _partial_sort(pts, index_map, c_ind, frm, to):
    """
    Partially sorts the given array by splitting it in an left sub-array which contains all
    elements smaller-equal than the median and a right sub-array which contains all elements
    greater-equal than the median.
    By construction, it is ensured that all elements from the left sub-array (only!), which are
    equal to the median, occur at the very end of the sub-array, just neighboring the median
    element.
    """
    # find median with adapted Hoare's and Wirth's method
    median_ind = (frm + to) // 2
    left = frm
    right = to - 1
    while left < right:
        # extract pivot value
        median_pivot = pts[index_map[median_ind], c_ind]
        # swap pivot node to beginning of relevant range
        h = index_map[left]
        index_map[left] = index_map[median_ind]
        index_map[median_ind] = h
        i = left + 1
        j = right
        while i <= j:
            # invariant: for all r with left+1 <= r < i: arr[r] < median_pivot
            #        and for all s with j < s <= right: median_pivot <= arr[s]
            while i <= right and pts[index_map[i], c_ind] < median_pivot:  i += 1
            # now holds: either i > right or median_pivot <= arr[i]
            while j > left and median_pivot <= pts[index_map[j], c_ind]:   j -= 1
            # now holds: either j <= left or arr[j] < median_pivot
            if i < j:
                # i.e. (i <= right and j > left) and (median_pivot <= arr[i] and arr[j] < median_pivot)
                # swap elements
                h = index_map[i]
                index_map[i] = index_map[j]
                index_map[j] = h
                i += 1
                j -= 1
                # invariant is reestablished
        # here we have j+1 == i and invariant, i.e.
        #       for all r with left+1 <= r <= j: arr[r] < median_pivot
        #   and for all s with j < s <= right: median_pivot <= arr[s]

        # reinsert pivot node at its correct place
        h = index_map[left]
        index_map[left] = index_map[j]
        index_map[j] = h

        if j < median_ind:    left = i
        elif j > median_ind:  right = j-1
        else:
            # j == medianIndex, i.e. we actually found median already and have it at the right place and
            # also the correct order of the sub arrays
            break


# -----------------------------------------------------------------------------

def _print_tree(tree, ind, pts):
    """
    Auxiliary function that traverses tree and prints tree nodes in infix order.
    Prints node index, its coordinates and then the indices of left and right child,
    nodes as well as index of parent node.
    """
    if tree[ind,0] >= 0:
        _print_tree(tree, tree[ind,0], pts)
    print('%4d: (%7.2f, %6.2f)  [lft: %4d, rgt: %4d, par: %4d]'
          % (ind, pts[ind,0], pts[ind,1], tree[ind,0], tree[ind,1], tree[ind,2]))
    if tree[ind,1] >= 0:
        _print_tree(tree, tree[ind,1], pts)


# -----------------------------------------------------------------------------

@njit
def prepare_search(radius, tree, pts):
    """
    Creates a radius search 'object' that can be used to retrieve points from
    the kd-tree given by tuple (tree, pts), refer to create_kdtree() function.
    
    This preparation step consists of creating two reusable arrays, that will
    contain the indices of the points that lie within the search radius around
    the search point and their square distances from it.
    
    The resulting tuple contains all information that is required to perform a
    radius search around a specific point.

    Parameters
    ----------
    radius : float
        The search radius to be used (using Euclidean norm).
    tree : numpy ndarray
        The index description of the kd-tree.
    pts : numpy ndarray
        The original point array.

    Returns
    -------
    res_index : numpy ndarray
        Will be used to store the indices of the retrieved point.
    res_sqr_dist : numpd ndarray
        Will be used to store the respective square distances.
    tree : numpy ndarray
        The index description of the kd-tree.
    pts : numpy ndarray
        The original point array.
    radius_sqr : float
        The square of the specified search radius.
    """

    num = len(pts)

    # encode number of valid res_index array elements at index -1
    res_index = np.full(num+1, -1, dtype=np.int32)
    res_sqr_dist = np.full(num, 999999.9, dtype=np.float64)
    
    radius_sqr = radius**2

    return res_index, res_sqr_dist, tree, pts, radius_sqr


# -----------------------------------------------------------------------------

@njit
def radius_search(search_pt, res_index, res_sqr_dist, tree, pts, radius_sqr):
    """
    Performs a radius search around the point `search_pt`. The remaining arguments
    that are passed to this function is the data tuple returned from prepare_search().
    The retrieved sample points are in general collected in an unordered way.
    The number of valid array entries is stored in array element `res_index[-1]`.
    
    The results are stored in the reusable arrays `res_index` and `res_sqr_dist` and
    thus are only valid before the next invocation of radius_search().
    
    Take care when using radius_search() in a multithreaded environment. Nevertheless,
    if each thread uses its own radius search 'object' (based on the same underlying
    kd-tree), parallel processing should be possible.

    Parameters
    ----------
    search_pt : numpy ndarray
        The coordinates of the search point.
    res_index : numpy ndarray
        The index array that specifies the points that lie within the search radius.
    res_sqr_dist : numpy ndarray
        The corresponding square distance to the search point.
    tree : numpy ndarray
        The index description of the kd-tree.
    pts : numpy ndarray
        The original point array.
    radius_sqr : float
        The square of the specified search radius.

    Returns
    -------
    None.
    """

    # reset number of valid res_index array elements
    res_index[-1] = 0
    _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[-1,0], search_pt, 0)


@njit
def _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, node_ind, coor, c_ind):
    """ The recursive kd-tree radius search implementation. """
    if coor[c_ind] < pts[node_ind,c_ind]:
        # go to the left side
        if tree[node_ind,0] >= 0:
            _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,0], coor, 1-c_ind)

        # check whether further tests are required
        if (coor[c_ind] - pts[node_ind,c_ind])**2 <= radius_sqr:
            # check this node against search radius
            sqr_dist = (coor[0]-pts[node_ind,0])**2 + (coor[1]-pts[node_ind,1])**2
            if sqr_dist <= radius_sqr:
                _append(res_sqr_dist, sqr_dist, res_index, node_ind)

            # check also nodes on the right side of the hyperplane
            if tree[node_ind,1] >= 0:
                _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,1], coor, 1-c_ind)

    else:
        # go the the right side
        if tree[node_ind,1] >= 0:
            _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,1], coor, 1-c_ind)

        # check whether further tests are required
        if (coor[c_ind] - pts[node_ind,c_ind])**2 <= radius_sqr:
            # check this node against search radius
            sqr_dist = (coor[0]-pts[node_ind,0])**2 + (coor[1]-pts[node_ind,1])**2
            if sqr_dist <= radius_sqr:
                _append(res_sqr_dist, sqr_dist, res_index, node_ind)

            # check also nodes on the left side of the hyperplane
            if tree[node_ind,0] >= 0:
                _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,0], coor, 1-c_ind)


@njit
def _append(res_sqr_dist, sqr_dist, res_index, node_ind):
    """
    Appends the sample point with index `node_ind` that has a distance of `sqr_dist`
    from the search point to the result arrays.
    """
    cur_len = res_index[-1]
    res_sqr_dist[cur_len] = sqr_dist
    res_index[cur_len] = node_ind
    # increase number of valid elements
    res_index[-1] += 1
    