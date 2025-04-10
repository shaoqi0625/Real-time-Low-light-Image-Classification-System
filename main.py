import os
import torch
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
        '''
        """运行整个 Pipeline，包括去噪、增强和分类。"""
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
        for image_path in Path(self.enhance_params["input_dir"]).rglob("*.*"):
            # 直接调用 lowlight 方法，lowlight 方法内部已经处理了路径和保存逻辑
            lowlight(str(image_path), self.enhance_params["output_dir"])
        print("图像增强完成！")
        '''
        
        # 3. 图像分类
        print("开始图像分类...")
        self.classify_images()
        print("图像分类完成！")

    def classify_images(self):
        """使用预训练模型对图像进行分类，并计算分类正确率及其他评价指标。"""
        model_path = self.classify_params["model_path"]
        input_dir = self.classify_params["input_dir"]
        input_size = self.classify_params.get("input_size", 260)
        device = self.classify_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # 显式设置 weights_only=False
        num_classes = self.classify_params.get("num_classes", 90)  # 根据实际类别数修改
        model = efficientnet_b2(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
        model = model.to(device)
        model.eval()

        # 定义图像预处理
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 获取类别映射（假设模型训练时类别顺序为 class_names）
        class_names = self.classify_params.get("class_names", [])  # 例如 ['cat', 'dog', 'elephant', ...]
        class_to_index = {name: idx for idx, name in enumerate(class_names)}

        # 初始化统计变量
        total_images = 0
        correct_predictions = 0
        all_true_labels = []
        all_predicted_labels = []

        # 遍历输入目录中的图像
        for image_path in Path(input_dir).rglob("*.*"):
            # 获取真实标签（文件夹名称即为标签）
            folder_name = image_path.parent.name  # 文件夹名称是动物种类标签
            true_label = class_to_index.get(folder_name, -1)  # 将文件夹名称映射为数字标签
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

            # 记录真实标签和预测标签
            all_true_labels.append(true_label)
            all_predicted_labels.append(predicted.item())

            # 比较预测结果与真实标签
            if predicted.item() == true_label:
                correct_predictions += 1
            total_images += 1

        # 计算分类正确率
        accuracy = 100.0 * correct_predictions / total_images if total_images > 0 else 0.0

        # 计算其他评价指标
        precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

        # 输出评价指标
        print(f"分类正确率: {accuracy:.2f}%")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1 分数: {f1:.4f}")
        print("混淆矩阵:")
        print(conf_matrix)


if __name__ == "__main__":
    # 配置参数
    denoise_params = {
        "input_dir": "dataset/animals_low_light/animals_low_light",
        "output_dir": "dataset/animals_denoised",
        "sigma_psd": 0.1
    }

    enhance_params = {
        "input_dir": "dataset/animals_denoised",
        "output_dir": "dataset/animals_enhanced"
    }

    classify_params = {
        "input_dir": "dataset/animals_low_light/animals_low_light",
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