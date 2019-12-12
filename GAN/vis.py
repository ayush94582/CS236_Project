# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left

        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

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



def render_3d_animation(poses, output, traj=False, depth=False, limit=1, size=6, norm=True):
    poses = np.reshape(poses[0,:limit, :], (1, limit, 17, 3))
    if depth:
        mat = np.zeros((3,3))
        c = 0.85
        f1 = 1145 * c
        f2 = 1143 * c
        mat[0,0] = f1
        mat[1,1] = f2
        mat[2,2] = 1
        mat[0,2] = 500
        mat[1,2] = 500
        pos2d = 1000*poses[0,:,:,:2]
        posz = 3 + 6 * poses[0,:,:,2]
        pos2d_cat = np.concatenate((pos2d, np.ones((limit, 17, 1))), axis=2)
        pos3d = np.zeros((1, limit, 17, 3))
        for i in range(limit):
            for j in range(17):
                pos3d[0,i,j,:] = posz[i,j] * np.matmul(np.linalg.inv(mat), pos2d_cat[i,j,:])
        poses = pos3d

    if traj:
        for i in range(limit):
            for j in range(1,17):
                poses[0,i,j,:] += poses[0,i,0,:]
    if norm:
        for i in range(limit):
            for j in range(1,17):
                poses[0,i,j,:] = poses[0,i,j,:] - poses[0,i,0,:]
            poses[0,i,0,:] = 0
    fps = 20
    bitrate = 3000
    azim = 20
    h36m_skeleton = Skeleton(parents=[-1,
                                  0,
                                  1,
                                  2,
                                  3,
                                  4,
                                  0,
                                  6,
                                  7,
                                  8,
                                  9,
                                  0,
                                  11,
                                  12,
                                  13,
                                  14,
                                  12,
                                  16,
                                  17,
                                  18,
                                  19,
                                  20,
                                  19,
                                  22,
                                  12,
                                  24,
                                  25,
                                  26,
                                  27,
                                  28,
                                  27,
                                  30],
                         joints_left=[6,
                                      7,
                                      8,
                                      9,
                                      10,
                                      16,
                                      17,
                                      18,
                                      19,
                                      20,
                                      21,
                                      22,
                                      23],
                         joints_right=[1,
                                       2,
                                       3,
                                       4,
                                       5,
                                       24,
                                       25,
                                       26,
                                       27,
                                       28,
                                       29,
                                       30,
                                       31])
    h36m_skeleton.remove_joints(
        [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
    h36m_skeleton._parents[11] = 8
    h36m_skeleton._parents[14] = 8
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))

    ax_3d = []
    lines_3d = []
    if norm:
        radius = 1.7
        poses = list(poses)
        for index in range(len(poses)):
            ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
            ax.view_init(elev=15., azim=azim)
            ax.set_xlim3d([-radius/2, radius/2])
            ax.set_zlim3d([-radius, radius])
            ax.set_ylim3d([-radius/2, radius/2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 7.5
            ax_3d.append(ax)
            lines_3d.append([])
    else:
        radius = 4
        poses = list(poses)
        for index in range(len(poses)):
            ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
            ax.view_init(elev=15., azim=70)
            ax.set_xlim3d([-radius/2, radius/2])
            ax.set_zlim3d([3, 9])
            ax.set_ylim3d([-radius/2, radius/2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 4.5
            ax_3d.append(ax)
            lines_3d.append([])
    initialized = False
    image = None
    lines = []
    points = None
    parents = h36m_skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        # Update 2D poses
        if not initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z'))
            initialized = True
        else:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j -
                                1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j -
                                1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties(
                        [pos[j, 2], pos[j_parent, 2]], zdir='z')

        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(
        fig,
        update_video,
        frames=np.arange(
            0,
            limit),
        interval=1000 /
        fps,
        repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError(
            'Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


