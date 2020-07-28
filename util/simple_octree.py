"""
Octree implementation
from https://github.com/jcummings2/pyoctree/blob/master/octree.py
"""
# From: https://code.google.com/p/pynastran/source/browse/trunk/pyNastran/general/octree.py?r=949
#       http://code.activestate.com/recipes/498121-python-octree-implementation/

# UPDATED:
# Is now more like a true octree (ie: partitions space containing objects)

# Important Points to remember:
# The OctNode positions do not correspond to any object position
# rather they are seperate containers which may contain objects
# or other nodes.

# An OctNode which which holds less objects than MAX_OBJECTS_PER_CUBE
# is a LeafNode; it has no branches, but holds a list of objects contained within
# its boundaries. The list of objects is held in the leafNode's 'data' property

# If more objects are added to an OctNode, taking the object count over MAX_OBJECTS_PER_CUBE
# Then the cube has to subdivide itself, and arrange its objects in the new child nodes.
# The new octNode itself contains no objects, but its children should.


import numpy as np


class OctNode(object):
    """
    New Octnode Class, can be appended to as well i think
    """
    def __init__(self, position, half_dimension, depth):
        """
        OctNode Cubes have a position and size
        position is the center of this node
        Branches (or children) follow a predictable pattern to make accesses simple.
        Here, - means less than 'origin' in that dimension, + means greater than.
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        """
        self.position = position
        self.half_dimension = half_dimension
        self.depth = depth

        # All OctNodes will be leaf nodes at first
        # Then subdivided later as more objects get added
        self.isLeafNode = True

        # store our object, typically this will be one, but maybe more
        self.obj_nums = []

        # might as well give it some emtpy branches while we are here.
        self.branches = [None, None, None, None, None, None, None, None]

        # The cube's bounding coordinates
        self.lower = position - half_dimension
        self.upper = position + half_dimension

    def get_obj_nums(self, obj_data_sets, half_search_offset):
        bounding_min = self.position - self.half_dimension - half_search_offset
        bounding_max = self.position + self.half_dimension + half_search_offset
        for obj_data in obj_data_sets:
            self.obj_nums.append(np.count_nonzero(np.asarray([(obj_data <= bounding_max).all(1),
                                               (obj_data >= bounding_min).all(1)]).all(0)))
        return self.obj_nums

    def devide_node(self, obj_data_sets, half_search_offset, max_obj_num):
        """Private version of insertNode() that is called recursively"""
        # we're inserting a single object, so if we reach an empty node, insert it here
        # Our new node will be a leaf with one object, our object
        # More may be added later, or the node maybe subdivided if too many are added
        # Find the Real Geometric centre point of our new node:
        # Found from the position of the parent node supplied in the arguments
        self.isLeafNode = False
        pos = self.position

        # offset is halfway across the size allocated for this node
        offset = self.half_dimension / 2
        new_center = np.zeros([8, 3], dtype=np.float64)
        for branch in range(8):
            if branch == 0:
                new_center[0, :] = pos - offset
            elif branch == 1:
                new_center[1, 0] = pos[0] - offset
                new_center[1, 1] = pos[1] - offset
                new_center[1, 2] = pos[2] + offset
            elif branch == 2:
                new_center[2, 0] = pos[0] - offset
                new_center[2, 1] = pos[1] + offset
                new_center[2, 2] = pos[2] - offset
            elif branch == 3:
                new_center[3, 0] = pos[0] - offset
                new_center[3, 1] = pos[1] + offset
                new_center[3, 2] = pos[2] + offset
            elif branch == 4:
                new_center[4, 0] = pos[0] + offset
                new_center[4, 1] = pos[1] - offset
                new_center[4, 2] = pos[2] - offset
            elif branch == 5:
                new_center[5, 0] = pos[0] + offset
                new_center[5, 1] = pos[1] - offset
                new_center[5, 2] = pos[2] + offset
            elif branch == 6:
                new_center[6, 0] = pos[0] + offset
                new_center[6, 1] = pos[1] + offset
                new_center[6, 2] = pos[2] - offset
            elif branch == 7:
                new_center[7, :] = pos + offset

        for branch in range(8):
            self.branches[branch] = OctNode(new_center[branch, :], offset, self.depth + 1)
            for index, tmp_offset in enumerate(list(half_search_offset)):
                self.branches[branch].get_obj_nums(
                    [obj_data_sets[2 * index], obj_data_sets[2 * index + 1]], tmp_offset)
            print(str(self.branches[branch]))
            if (np.asarray(self.branches[branch].obj_nums) > max_obj_num).any(0):
                self.branches[branch].devide_node(obj_data_sets, half_search_offset, max_obj_num)

    def gather_leaf_node(self, node_output):
        for branch in self.branches:
            if branch is None:
                continue
            if not branch.isLeafNode:
                branch.gather_leaf_node(node_output)
            if branch.isLeafNode:
                node_output.append(branch)

    def __str__(self):
        return u"position: {0}, size: {1}, depth: {2} leaf: {3} obj_num: {4}".format(
            self.position, self.half_dimension*2, self.depth, self.isLeafNode, self.obj_nums
        )


class Octree(object):
    """
    The octree itself, which is capable of adding and searching for nodes.
    """
    def __init__(self, world_size, origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                 max_type="nodes", max_value=200000):
        """
        Init the world bounding root cube
        all world geometry is inside this
        it will first be created as a leaf node (ie, without branches)
        this is because it has no objects, which is less than MAX_OBJECTS_PER_CUBE
        if we insert more objects into it than MAX_OBJECTS_PER_CUBE, then it will subdivide itself.
        """
        self.root = OctNode(origin + world_size/2.0, world_size/2.0, 0)
        self.worldSize = world_size
        self.limit_nodes = (max_type == "nodes")
        self.limit = max_value

    def divide_octree(self, obj_data_sets, half_search_offset, max_obj_num):
        """
        Add the given object to the octree if possible
        Parameters
        ----------
        position : array_like with 3 elements
            The spatial location for the object
        objData : optional
            The data to store at this position. By default stores the position.
            If the object does not have a position attribute, the object
            itself is assumed to be the position.
        Returns
        -------
        node : OctNode or None
            The node in which the data is stored or None if outside the
            octree's boundary volume.
        """
        for index, offset in enumerate(list(half_search_offset)):
            self.root.get_obj_nums([obj_data_sets[2*index], obj_data_sets[2*index+1]], offset)
        if (np.asarray(self.root.obj_nums) > max_obj_num).any(0):
            self.root.devide_node(obj_data_sets, half_search_offset, max_obj_num)

    def gather_leaf_node(self):
        node_output = []
        if self.root.isLeafNode:
            node_output.append(self.root)
        else:
            self.root.gather_leaf_node(node_output)
        return node_output


