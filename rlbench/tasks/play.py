# global
import os
import math
import trimesh
import ivy_mech
import numpy as np
from typing import List
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.objects.object import Object
from pyrep.errors import ConfigurationPathError
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects.proximity_sensor import ProximitySensor


class Play(Task):

    # Private #
    # --------#

    def _create_walls(self):
        workspace = Shape('workspace')
        self._workspace_pos = workspace.get_position()
        workspace_z = self._workspace_pos[-1]
        bbx = workspace.get_bounding_box()
        self._workspace_size = np.array([bbx[1] - bbx[0], bbx[3] - bbx[2], bbx[5] - bbx[4]])
        wall_height = 0.5
        wall_xy_positions = [self._workspace_pos[0:2] +
                             np.array([i*self._workspace_size[0]/2, j*self._workspace_size[1]/2])
                             for i, j in [[0, -1], [0, 1], [-1, 0], [1, 0]]]
        wall_positions = [np.concatenate((pos, np.array([workspace_z + wall_height/2])), 0)
                          for pos in wall_xy_positions]
        wall_orientations = [np.zeros(3), np.array([0., 0., -math.pi]), np.array([0., 0., -math.pi/2]),
                             np.array([0., 0., math.pi/2])]
        wall_sizes = [np.array([self._workspace_size[0], 0.01, wall_height]),
                      np.array([self._workspace_size[0], 0.01, wall_height]),
                      np.array([self._workspace_size[1], 0.01, wall_height]),
                      np.array([self._workspace_size[1], 0.01, wall_height])]
        self._x_lim = self._workspace_size[0]/2
        self._y_lim = self._workspace_size[1]/2
        self._walls = list()
        for wall_pos, wall_ori, wall_size in zip(wall_positions, wall_orientations, wall_sizes):
            wall = Shape.create(PrimitiveShape.CUBOID, wall_size.tolist(), position=wall_pos,
                                orientation=wall_ori, static=True, renderable=False)
            wall.set_transparency(0.)
            self._walls.append(wall)

    def _add_walls(self):
        for wall in self._walls:
            wall.set_respondable(True)
            wall.set_collidable(True)
            wall.set_renderable(True)

    def _remove_walls(self):
        for wall in self._walls:
            wall.set_respondable(False)
            wall.set_collidable(False)
            wall.set_renderable(False)

    def _remove_out_of_scope_objects(self):
        to_remove = list()
        obj_diags = list()
        for obj in self._objects:
            bbx = obj.get_bounding_box()
            x_diff = bbx[1] - bbx[0]
            y_diff = bbx[3] - bbx[2]
            z_diff = bbx[5] - bbx[4]
            diag = (x_diff**2 + y_diff**2 + z_diff**2)**0.5
            obj_diags.append(diag)
        for idx, (obj, obj_diag) in enumerate(zip(self._objects, obj_diags)):
            obj_pos = obj.get_position()
            x_out_of_bounds = np.abs(obj_pos[0] - self._workspace_pos[0] + obj_diag/2) + 0.01 > self._x_lim
            y_out_of_bounds = np.abs(obj_pos[1] + obj_diag/2) + 0.01 > self._y_lim
            if x_out_of_bounds or y_out_of_bounds:
                obj.remove()
                to_remove.append(idx)
        for i, idx in enumerate(to_remove):
            del self._objects[idx - i]

    @staticmethod
    def _normal_to_rot_mat(normal):
        z_axis = -normal
        x_axis = np.r_[1.0, 0.0, 0.0]
        if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
            x_axis = np.r_[0.0, 1.0, 0.0]
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)
        return np.vstack((x_axis, y_axis, z_axis)).T

    def _assemble_scene(self):
        while len(self._objects) == 0:
            self._add_walls()
            num_objects = np.random.randint(self._min_num_objects, self._max_num_objects)
            for i in range(num_objects):
                obj_type = np.random.choice(self._permitted_shapes)
                position = self._workspace_pos + \
                           np.random.uniform(-self._workspace_size / 2,
                                             self._workspace_size / 2) + \
                           np.array([0., 0., np.random.uniform(0.3, 0.7)])
                shape = Shape.create(obj_type, np.random.uniform(0.02, 0.2, size=3).tolist(),
                                     position=position, orientation=(np.random.uniform(size=3)-0.5)*2*math.pi,
                                     color=np.random.uniform(size=3).tolist())
                shape.set_collidable(True)
                shape.set_parent(self._base)
                self._objects.append(shape)
                for _ in range(self._drop_interval):
                    self.pyrep.step()
            self._object_names.clear()
            self._object_names = [obj.get_name() for obj in self._objects]
            self._remove_walls()
            for _ in range(self._settle_time):
                self.pyrep.step()
            for obj in self._objects:
                obj.set_dynamic(False)
            self._remove_out_of_scope_objects()

    def _get_combined_objects_mesh(self):
        scene_viz = self.pyrep.get_scene_viz()
        vertices_list = list()
        lengths_list = list()
        indices_list = list()
        vertex_normals_list = list()
        for j, name in enumerate(scene_viz.names):
            if name.split('/')[0] not in self._object_names:
                continue
            vertices_list.append(scene_viz.vertices[j])
            lengths_list.append(scene_viz.vertices[j].shape[0])
            indices_list.append(scene_viz.indices[j])
            vertex_normals_list.append(scene_viz.normals[j])
        vertices = np.concatenate(vertices_list, 0)
        cum_lengths = np.cumsum([0] + lengths_list[:-1]).tolist()
        indices = np.concatenate([ids + cum_len for ids, cum_len in zip(indices_list, cum_lengths)], 0)
        vertex_normals = np.concatenate(vertex_normals_list, 0)
        return trimesh.Trimesh(vertices, indices, vertex_normals=vertex_normals)

    # Public #
    # -------#

    def init_task(self) -> None:

        np.random.seed(0)
        self._base = self.get_base()
        self._min_num_objects = 5
        self._max_num_objects = 15
        self._permitted_shapes = [PrimitiveShape.CUBOID]
        # noinspection PyProtectedMember
        self._create_walls()
        self._objects = list()
        self._object_names = list()
        self._drop_interval = 5
        self._settle_time = 100
        self._num_robot_poses_per_scene = 1
        self._finger_depth = 0.05
        self._success_sensor = ProximitySensor('success')

        # waypoints
        # Note: this CANNOT be called waypoints, to avoid confusion with RLBench internal variable
        self._waypoints_list = list()
        for i in range(self._num_robot_poses_per_scene):
            waypoint = Dummy.create()
            waypoint.set_position(self._workspace_pos + np.array([0., 0., 0.5]))
            waypoint.set_name('waypoint{}'.format(i))
            waypoint.set_parent(self._base)
            self._waypoints_list.append(waypoint)
        self._success_sensor.set_position(self._waypoints_list[-1].get_position())

        # success cond
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), self._success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        print('assembling scene...')
        self._assemble_scene()
        print('assembled scene.')
        tmesh = self._get_combined_objects_mesh()
        num_waypoints_Placed = 0
        print('finding valid waypoints...')
        while num_waypoints_Placed < self._num_robot_poses_per_scene:
            ok = False
            while not ok:
                idx_to_try = np.random.randint(tmesh.vertices.shape[0])
                vertex = tmesh.vertices[idx_to_try]
                # noinspection PyPropertyAccess
                vertex_normal = tmesh.vertex_normals[idx_to_try]
                ok = vertex_normal[2] > -0.1  # make sure the normal is poitning upwards
            grasp_depth = np.random.uniform(-0.1 * self._finger_depth, 1.1 * self._finger_depth)
            # noinspection PyUnboundLocalVariable
            point = vertex + vertex_normal * grasp_depth
            rot_mat = self._normal_to_rot_mat(vertex_normal)
            quat = ivy_mech.rot_mat_to_quaternion(rot_mat)
            quat = quat / np.linalg.norm(quat)
            gripper = np.array([1.])
            action = np.concatenate((point, quat, gripper), 0)
            try:
                print('trying path...')
                self.robot.arm.get_path(point, quaternion=quat, distance_threshold=0.1)
            except ConfigurationPathError:
                print('path failed.')
                continue
            print('found waypoint {}'.format(num_waypoints_Placed))
            self._waypoints_list[num_waypoints_Placed].set_position(point)
            self._waypoints_list[num_waypoints_Placed].set_quaternion(quat)
            num_waypoints_Placed += 1
        self._success_sensor.set_position(self._waypoints_list[-1].get_position())
        print('initialized episode')
        return ['']

    def variation_count(self):
        return 1

    def base_rotation_bounds(self):
        return (0., 0., 0.), (0., 0., 0.)

    def load(self) -> Object:
        if Object.exists(self.get_name()):
            return Dummy(self.get_name())
        ttm_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object

    def cleanup(self):
        for obj in self._objects:
            obj.remove()
        self._objects.clear()

    def get_low_dim_state(self):
        state = []
        for obj in self._objects:
            state.extend(np.array(obj.get_pose()))
        return np.array(state).flatten()
