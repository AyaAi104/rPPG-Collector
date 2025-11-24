"""
图像序列转视频脚本 / Image Sequence to Video Converter
用于 rPPG 工具箱的数据预处理 / For rPPG Toolbox Data Preprocessing

使用方法 / Usage:
python images_to_video.py --input_folder /path/to/images --output_video output.mp4 --fps 50
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path


def convert_images_to_video(input_folder, output_video, fps=50, image_extension='.png'):
    """
    将图像序列转换为视频文件 / Convert image sequence to video file

    参数 / Parameters:
    - input_folder: 包含图像的文件夹路径 / Path to folder containing images
    - output_video: 输出视频文件路径 / Output video file path
    - fps: 帧率 / Frame rate (default: 50)
    - image_extension: 图像文件扩展名 / Image file extension
    """

    # 获取所有图像文件并排序 / Get all image files and sort them
    image_files = sorted([f for f in os.listdir(input_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if not image_files:
        raise ValueError(f" No image files found in {input_folder}")

    print(f"找到 {len(image_files)} 张图像 / Found {len(image_files)} images")
    print(f"第一张: {image_files[0]}, 最后一张: {image_files[-1]}")
    print(f"First: {image_files[0]}, Last: {image_files[-1]}")

    # 读取第一张图像以获取尺寸 / Read first image to get dimensions
    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path)

    if first_image is None:
        raise ValueError(f"Converter:  Cannot read image: {first_image_path}")

    height, width, channels = first_image.shape
    print(f"Converter: Image dimensions: {width}x{height}")

    # 创 Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID' / or use 'XVID'
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Converter:  Cannot create video writer")

    #  Write frames
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Converter:  / Warning: Cannot read {image_file}, skipping")
            continue

        out.write(frame)

        # 显示进度 / Show progress
        if (i + 1) % 50 == 0:
            print(f"Converter: / Processed {i + 1}/{len(image_files)} frames")

    # 释放资源 / Release resources
    out.release()

    print(f"\n✓ Converter:  / Video created successfully: {output_video}")
    print(f"  - Converter:  / Total frames: {len(image_files)}")
    print(f"  - Converter:  / Frame rate: {fps} FPS")
    print(f"  - Converter:  / Duration: {len(image_files) / fps:.2f} 秒 / seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Converter:  / Convert image sequence to video'
    )
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Converter:  / Input folder containing images')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Converter:  / Output video file path')
    parser.add_argument('--fps', type=int, default=50,
                        help='Converter:  / Video frame rate (default: 50)')

    args = parser.parse_args()

    # / Verify input folder exists
    if not os.path.exists(args.input_folder):
        raise ValueError(f"Converter:  / Input folder does not exist: {args.input_folder}")

    # / Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" Converter: / Created output directory: {output_dir}")

    # 转换 / Convert
    convert_images_to_video(args.input_folder, args.output_video, args.fps)


if __name__ == "__main__":
    main()