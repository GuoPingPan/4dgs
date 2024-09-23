import open3d as o3d

# 创建一个可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 读取 PLY 文件
ply_file_path = 'path/to/your/file.ply'  # 替换为你的 PLY 文件路径
ply_file_path = "/home/a4090/4d-gaussian-splatting/output/N3V/flame_steak/dynamic/ours_-1/mesh/fuse_1_post.ply"
pcd = o3d.io.read_point_cloud(ply_file_path)

# 添加点云对象到可视化窗口
vis.add_geometry(pcd)

# 获取视图控制器
view_ctl = vis.get_view_control()

# 设置相机的视点、目标点和上向量
eye = [1.1, 0, 0]  # 相机的位置
lookat = [1, 0, 0]  # 相机注视的目标
up = [0, 0, 1]  # 上向量设置为 Y 轴

# 更新视图控制器
view_ctl.set_lookat(lookat)
# view_ctl.set_front([eye[i] - lookat[i] for i in range(3)])
view_ctl.set_front([-1, -1, 0])
view_ctl.set_up(up)
# view_ctl.set_zoom(0.25)  # 增加视图缩放级别，使得对象更大

# 启动可视化窗口
vis.run()

# 关闭可视化窗口
vis.destroy_window()
