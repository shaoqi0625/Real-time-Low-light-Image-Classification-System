import os
import torch
import time  # 导入 time 模块
from pathlib import Path
from Data_preprocess import denoise_images
from ZeroDCE import lowlight
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

class ImageProcessingPipeline:
    def __init__(self, denoise_params, enhance_params, classify_params):
        """
        初始化图像处理 Pipeline。

        Args:
            denoise_params (dict): 去噪参数，包括输入和输出路径等。
            enhance_params (dict): 图像增强参数，包括输入和输出路径等。
            classify_params (dict): 图像分类参数，包括模型路径、输入大小等。
        """
        self.denoise_params = denoise_params
        self.enhance_params = enhance_params
        self.classify_params = classify_params

    def run(self):
        """运行整个 Pipeline，包括去噪、增强和分类。"""
        start_time = time.time()  # 记录开始时间

        # 1. 图像去噪
        print("开始图像去噪...")
        denoise_images(
            self.denoise_params["input_dir"],
            self.denoise_params["output_dir"],
            sigma_psd=self.denoise_params.get("sigma_psd", 0.1)
        )
        print("图像去噪完成！")
        
        # 2. 图像增强
        print("开始图像增强...")
        input_dir = Path(self.enhance_params["input_dir"])
        output_dir = Path(self.enhance_params["output_dir"])

        for image_path in input_dir.rglob("*.*"):
            # 调整路径以兼容 ZeroDCE.py 的逻辑
            relative_path = image_path.relative_to(input_dir)
            adjusted_input_path = Path("dataset/animals_denoised") / relative_path
            adjusted_input_path.parent.mkdir(parents=True, exist_ok=True)
            adjusted_input_path.write_bytes(image_path.read_bytes())  # 复制文件到调整后的路径

            # 调用 lowlight 方法
            lowlight(str(adjusted_input_path), str(output_dir))
        print("图像增强完成！")
        
        # 3. 图像分类
        print("开始图像分类...")
        self.classify_images()
        print("图像分类完成！")

        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time
        print(f"整个 Pipeline 运行时间: {total_time:.2f} 秒")

    def classify_images(self):
        """使用预训练模型对图像进行分类，并输出分类的类别。"""
        model_path = self.classify_params["model_path"]
        input_dir = self.classify_params["input_dir"]
        input_size = self.classify_params.get("input_size", 260)
        device = self.classify_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        num_classes = self.classify_params.get("num_classes", 90)
        model = efficientnet_b2(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # 定义图像预处理
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 获取类别映射
        class_names = self.classify_params.get("class_names", [])
        class_to_index = {name: idx for idx, name in enumerate(class_names)}

        # 遍历输入目录中的图像
        for image_path in Path(input_dir).rglob("*.*"):
            folder_name = image_path.parent.name
            true_label = class_to_index.get(folder_name, -1)
            if true_label == -1:
                print(f"未知类别标签: {folder_name}, 跳过图像 {image_path}")
                continue

            # 加载图像
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            # 进行分类
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)

            # 输出分类的类别
            predicted_class = class_names[predicted.item()]
            print(f"图像: {image_path.name}, 分类结果: {predicted_class}")


if __name__ == "__main__":
    # 配置参数
    denoise_params = {
        "input_dir": "testdata/lowlight",
        "output_dir": "testdata/denoised",
        "sigma_psd": 0.1
    }

    enhance_params = {
        "input_dir": "testdata/denoised",
        "output_dir": "testdata/enhanced"
    }

    classify_params = {
        "input_dir": "testdata/enhanced",
        "model_path": "EfficientNet/checkpoints/animal_classifier/final_model.pth",
        "input_size": 260,
        "num_classes": 90,  # 假设有 3 个类别
        "class_names": ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 
                        'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 
                        'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 
                        'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 
                        'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 
                        'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 
                        'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 
                        'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 
                        'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 
                        'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 
                        'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 
                        'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 
                        'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 
                        'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 
                        'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 
                        'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'],  # 类别名称列表
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 创建并运行 Pipeline
    pipeline = ImageProcessingPipeline(denoise_params, enhance_params, classify_params)
    pipeline.run()