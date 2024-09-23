import open3d as o3d
import numpy as np

class CustomVisualizer(o3d.visualization.VisualizerWithKeyCallback):
    def __init__(self):
        super().__init__()
        self.key_callbacks = {
            ord('W'): self.move_forward,
            ord('S'): self.move_backward,
            ord('A'): self.move_left,
            ord('D'): self.move_right,
            ord('Q'): self.rotate_left,
            ord('E'): self.rotate_right,
        }

    def key_callback(self, vis, key, action, mods):
        if action == o3d.visualization.KeyAction.PRESS:
            if key in self.key_callbacks:
                self.key_callbacks[key]()

    def move_forward(self):
        ctr = self.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic[:3, 3] += 0.1 * param.extrinsic[:3, 2]  # Move forward
        ctr.convert_from_pinhole_camera_parameters(param)
    
    def move_backward(self):
        ctr = self.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic[:3, 3] -= 0.1 * param.extrinsic[:3, 2]  # Move backward
        ctr.convert_from_pinhole_camera_parameters(param)
    
    def move_left(self):
        ctr = self.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic[:3, 3] -= 0.1 * param.extrinsic[:3, 0]  # Move left
        ctr.convert_from_pinhole_camera_parameters(param)
    
    def move_right(self):
        ctr = self.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic[:3, 3] += 0.1 * param.extrinsic[:3, 0]  # Move right
        ctr.convert_from_pinhole_camera_parameters(param)

    def rotate_left(self):
        ctr = self.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, 0.1])
        param.extrinsic[:3, :3] = rotation_matrix @ param.extrinsic[:3, :3]  # Rotate around Z-axis
        ctr.convert_from_pinhole_camera_parameters(param)
    
    def rotate_right(self):
        ctr = self.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, -0.1])
        param.extrinsic[:3, :3] = rotation_matrix @ param.extrinsic[:3, :3]  # Rotate around Z-axis
        ctr.convert_from_pinhole_camera_parameters(param)

def main(ply_file_path):
    vis = CustomVisualizer()
    vis.create_window()
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_file_path)
    vis.add_geometry(pcd)

    # 注册键盘回调函数
    vis.register_key_callback(ord('W'), vis.key_callback)
    vis.register_key_callback(ord('S'), vis.key_callback)
    vis.register_key_callback(ord('A'), vis.key_callback)
    vis.register_key_callback(ord('D'), vis.key_callback)
    vis.register_key_callback(ord('Q'), vis.key_callback)
    vis.register_key_callback(ord('E'), vis.key_callback)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    ply_file_path = "path_to_your_file.ply"  # 替换成你的 PLY 文件路径
    ply_file_path = "/home/a4090/4d-gaussian-splatting/output/N3V/flame_steak/dynamic/ours_-1/mesh/fuse_1_post.ply"
    main(ply_file_path)
