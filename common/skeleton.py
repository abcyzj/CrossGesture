import numpy as np

class Skeleton:
    def __init__(self, parents, joints_left, joints_right, bone_lengths):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._bone_lengths = bone_lengths
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children
    
    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def bone_lengths(self):
        return self._bone_lengths

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
