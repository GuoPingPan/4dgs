import open3d as o3d
import numpy as np

def print_camera_position(vis):
    """
    这个回调函数将在每次视图更新时被调用
    打印当前相机的位置和方向
    """
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    print("Camera position: ", camera_params.extrinsic[:3, 3])
    print("Camera look at: ", camera_params.extrinsic[:3, 2] + camera_params.extrinsic[:3, 3])
    return False  # 返回 False 表示不需要继续调用这个回调函数

def main(ply_file_path):
    # 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到窗口
    vis.add_geometry(pcd)

    # 获取视图控制器
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    # 设置相机参数
    # camera_params = ctr.convert_to_pinhole_camera_parameters()
    ctr.set_up([0, 0, 1])
    
    # # 设置相机的位置 (2, 2, 2)
    # camera_position = np.array([2, 2, 2])
    # ctr.camera_local_translate(2, 2, 2)
    
    # 设置目标位置（通常为点云的中心）
    look_at = np.array([0, 0, 0]).astype(np.float64)
    # vec = look_at - camera_position
    # vec /= np.linalg.norm(vec)
    # ctr.set_front(vec*5)  # 相机朝向
    # ctr.set_lookat(look_at)  # 相机朝向
    
    camera_params.extrinsic = np.array(
         [[ 1.        ,  0.        ,  0.        ,  2],
        [-0.        , -1.        , -0.        , 2],
        [-0.        , -0.        , -1.        , 2],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    
    print("Camera position: \n", np.array2string(camera_params.extrinsic, separator=', '))
    # 设置相机的“上”向量
    # up_vector = np.array([0, 0, 1]).astype(np.float64)
    
    # # 计算相机的旋转矩阵
    # forward_vector = (look_at - camera_position)
    # forward_vector /= np.linalg.norm(forward_vector)  # 归一化
    # right_vector = np.cross(up_vector, forward_vector)
    # right_vector /= np.linalg.norm(right_vector)  # 归一化
    # new_up_vector = np.cross(forward_vector, right_vector)
    
    # # 创建新的相机外参矩阵
    # extrinsic = np.eye(4)
    # extrinsic[:3, :3] = np.vstack([right_vector, new_up_vector, -forward_vector]).T
    # extrinsic[:3, 3] = -camera_position
    
    # # 更新视图控制器的相机参数
    # camera_params.extrinsic = extrinsic
    # ctr.convert_from_pinhole_camera_parameters(camera_params)
    
    # 注册相机位置回调函数
    # vis.register_animation_callback(print_camera_position)

    # 运行可视化窗口
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 替换成你的 PLY 文件路径
    ply_file_path = "/home/a4090/4d-gaussian-splatting/output/N3V/flame_steak/dynamic/ours_-1/mesh/fuse_1_post.ply"
    main(ply_file_path)
