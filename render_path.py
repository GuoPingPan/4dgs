import open3d as o3d
import numpy as np
import cv2
import os
from tqdm import tqdm
# from render_dynamic import do_system

def do_system(arg):
    print(f"==== running: {arg}")
    import ipdb; ipdb.set_trace()
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def render_frame(ply_file, camera_pos, look_at_point, output_image):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.background_color = [0.0, 0.0, 0.0]  # RGB 值，范围为 [0, 1]

    # 设置相机参数
    ctr = vis.get_view_control()
    ctr.set_up([0, 0, 1])
    ctr.set_lookat(look_at_point)  # 相机观察的目标点（世界原点）
    ctr.set_front(camera_pos - look_at_point)  # 相机朝向
    ctr.set_zoom(0.125)  # 相机朝向
    
    # 渲染
    vis.poll_events()
    vis.update_renderer()
    
    # 截取图像
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    
    # 保存图像
    cv2.imwrite(output_image, (image * 255)[..., ::-1].astype(np.uint8))
    
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
    ply_files = [f"output/N3V/flame_steak/dynamic/ours_-1/mesh/fuse_{i}.ply" for i in range(0, 10 + 1)]  # 根据需要设置时间范围
    image_folder = "images"
    os.makedirs(image_folder, exist_ok=True)

    # 圆心（经纬度点）
    look_at_point = np.array([0, 0, 0])

    # 设置相机的圆弧轨迹半径和角度范围
    radius = 100
    views_per_frame = 1
    num_frames = len(ply_files) * views_per_frame
    
    for i, ply_file in enumerate(tqdm(ply_files)):
        for j in range(views_per_frame):
            it = i*views_per_frame + j
            angle = np.pi * it / num_frames  + -np.pi/2 # 按时间点计算角度
            camera_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.])
            output_image = os.path.join(image_folder, f"frame_{it:04d}.png")
            print("Camera position:", camera_pos)
            render_frame(ply_file, camera_pos, look_at_point, output_image)

    # create_video_from_images(image_folder, "output_video.mp4")
    do_system(f"ffmpeg -y -r 30 -i {image_folder}/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output_video.mp4")

if __name__ == "__main__":
    main()
