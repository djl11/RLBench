import os
import time
import pickle
import numpy as np
import open3d as o3d


def spin_sim(vis, observations, first_pass):
    o3d_mesh = None
    for i, obs in enumerate(observations):
        names = obs.misc['scene_viz']['names']
        vertices = obs.misc['scene_viz']['vertices']
        indices = obs.misc['scene_viz']['indices']
        vertex_normals = obs.misc['scene_viz']['normals']
        colors = obs.misc['scene_viz']['colors']
        vert_list = list()
        color_list = list()
        ind_list = list()
        norm_list = list()
        total_vertices_so_far = 0
        for vert, ind, norm, col, name in zip(vertices, indices, vertex_normals, colors, names):
            if vert.shape[0] != norm.shape[0]:
                continue
            if name == 'ResizableFloor_5_25_visibleElement/2':
                continue
            vert_list.append(vert)
            color_list.append(np.ones_like(vert) * col[0:3])
            ind_list.append(ind + total_vertices_so_far)
            norm_list.append(norm)
            total_vertices_so_far += vert.shape[0]
        vertices = np.concatenate(vert_list, 0)
        vertex_colors = np.concatenate(color_list, 0)
        indices = np.concatenate(ind_list, 0)
        vertex_normals = np.concatenate(norm_list, 0)
        if vertices.shape[0] != vertex_normals.shape[0]:
            raise Exception('found {} vertices and {} vertex normals, but expected the number to be the same.'.format(
                indices.shape[0], vertex_normals.shape[0]
            ))
        if i == 0:
            o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                                 o3d.utility.Vector3iVector(indices))
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            vis.clear_geometries()
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.15, [0., 0., 0.]), first_pass)
            vis.add_geometry(o3d_mesh, first_pass)
            first_pass = False
        else:
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(indices)
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
            vis.update_geometry(o3d_mesh)
        vis.poll_events()
        time.sleep(0.05)


def main():
    path = '/media/djl11/Drive/QAIL/reach_target/variation0/episodes/episode0'
    misc_path = os.path.join(path, 'low_dim_obs.pkl')
    with open(misc_path, 'rb') as file:
        demo = pickle.load(file)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_pass = True
    while True:
        # noinspection PyProtectedMember
        spin_sim(vis, demo._observations, first_pass)
        first_pass = False


if __name__ == '__main__':
    main()
