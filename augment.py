import albumentations as A
import os
import cv2
import warnings
warnings.filterwarnings("ignore", message="Error fetching version info")
import matplotlib.pyplot as plt

class ImageAugmentor:
    def __init__(self, data_path, output_path, num_augmentations, transforms):
        self.data_path = data_path
        self.output_path = output_path
        self.transforms = transforms
        self.num_augmentations = num_augmentations
        self.count = 0

    def augment_single_img(self):
        folders = os.listdir(self.data_path)
        os.makedirs(self.output_path, exist_ok=True)
        for folder in folders:
            folder_path = os.path.join(self.data_path, folder)
            files = os.listdir(folder_path)
            save_path = os.path.join(self.output_path, folder)
            os.makedirs(save_path, exist_ok=True)
            index = len(os.listdir(save_path))
            for file in files:
                file_path = os.path.join(folder_path, file)
                image = cv2.imread(file_path)
                for num in range(self.num_augmentations):
                    index += 1
                    augmented = self.transforms(image=image)
                    filename = folder + "_" + str(index) + ".jpeg"
                    full_save_path = os.path.join(save_path, filename)
                    cv2.imwrite(full_save_path, augmented["image"])
                    self.count += 1
                    print(f"Save successfully augment of image ", filename)
        print("Completed augmentation!")

# transform = A.Compose([
#     A.Resize(64, 64),
#     A.ShiftScaleRotate(
#         shift_limit=0.05,     # tịnh tiến nhẹ
#         scale_limit=0.05,     # phóng nhỏ/xíu
#         rotate_limit=10,      # xoay nhẹ
#         border_mode=0,
#         p=0.6
#     ),
#     A.HorizontalFlip(p=0.2),  # chỉ dùng nếu hình flip vẫn giữ meaning
#     A.RandomElasticTransform(alpha=10, sigma=5, alpha_affine=5, p=0.25), # tăng đa dạng nét vẽ
# ])