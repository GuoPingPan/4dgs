import open3d as o3d
import numpy as np
import cv2
import os

def render_frame(ply_file, camera_pos, look_at_point, output_image):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    
    # 设置相机参数
    ctr = vis.get_view_control()
    ctr.set_lookat(look_at_point)  # 相机观察的目标点（世界原点）
    ctr.set_up([0, 1, 0])
    ctr.set_front(camera_pos - look_at_point)  # 相机朝向
    
    # 渲染
    vis.poll_events()
    vis.update_renderer()
    
    # 截取图像
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    
    # 保存图像
    cv2.imwrite(output_image, (image * 255).astype(np.uint8))
    
    vis.destroy_window()

def create_video_from_images(image_folder, output_video, frame_rate=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure images are in the correct order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def main():
    ply_files = [f"model_{i}.ply" for i in range(start_time, end_time + 1)]  # 根据需要设置时间范围
    image_folder = "images"
    os.makedirs(image_folder, exist_ok=True)

    # 圆心（经纬度点）
    look_at_point = np.array([0, 0, 0])

    # 设置相机的圆弧轨迹半径和角度范围
    radius = 10
    num_frames = len(ply_files)
    
    for i, ply_file in enumerate(ply_files):
        angle = 2 * np.pi * i / num_frames  # 按时间点计算角度
        camera_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 5])
        output_image = os.path.join(image_folder, f"frame_{i:04d}.png")
        render_frame(ply_file, camera_pos, look_at_point, output_image)

    create_video_from_images(image_folder, "output_video.mp4")

if __name__ == "__main__":
    main()
