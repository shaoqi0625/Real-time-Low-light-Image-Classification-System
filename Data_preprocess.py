import cv2
import numpy as np
import os
from pathlib import Path
from skimage import img_as_ubyte
from bm3d import bm3d

def denoise_images(input_dir, output_dir, sigma_psd=0.1):
    """
    对指定目录中的图片进行 BM3D 去噪处理，并将结果保存到新的目录中。

    Args:
        input_dir (str): 输入图片的根目录，每个子文件夹为一个类别。
        output_dir (str): 输出去噪图片的根目录，保持与输入目录相同的结构。
        sigma_psd (float): BM3D 去噪的噪声标准差，默认为 0.1。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历每个类别文件夹
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # 遍历类别文件夹中的每张图片
            for image_path in class_dir.glob("*.*"):
                # 读取图片
                noisy_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if noisy_img is None:
                    print(f"无法读取图片: {image_path}")
                    continue
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

                # 归一化到 [0, 1] 范围
                noisy_img_norm = noisy_img / 255.0

                # 进行 BM3D 去噪
                denoised_img_norm = bm3d(noisy_img_norm, sigma_psd=sigma_psd)

                # 确保去噪后的图像在 [0, 1] 范围内
                denoised_img_norm = np.clip(denoised_img_norm, 0, 1)

                # 还原到 8-bit 图像格式
                denoised_img = img_as_ubyte(denoised_img_norm)

                # 保存去噪后的图片
                output_image_path = output_class_dir / image_path.name
                cv2.imwrite(str(output_image_path), cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))

                # print(f"已处理图片: {image_path} -> {output_image_path}")

if __name__ == "__main__":
    # 输入和输出目录
    input_dir = "dataset/animals_low_light/animals_low_light"
    output_dir = "dataset/animals_denoised"

    # 调用去噪函数
    denoise_images(input_dir, output_dir, sigma_psd=0.1)

    print("所有图片已处理完成！")