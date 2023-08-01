import os
import cv2
import numpy as np
import albumentations as A
import uuid

def augment_images(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = os.listdir(images_folder)

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, image_file.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as file:
            lines = file.readlines()

        bboxes = []
        class_ids = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_ids.append(int(class_id))

        # Augment images with different transformations
        for i in range(9):  # 12 augmentations in total
            if i == 0:
                augmented_image = image.copy()  # Original version
                augmented_bboxes = bboxes.copy()
            else:
                transforms = []

                if i == 1:
                    transforms.append(A.ToGray(p=1.0))
                elif i == 2:
                    transforms.append(A.RandomBrightness(limit=0.1, p=1.0))
                elif i == 3:
                    transforms.append(A.RandomBrightness(limit=-0.1, p=1.0))
                elif i == 4:
                    transforms.append(A.Blur(blur_limit=2, p=1.0))
                elif i == 5:
                    transforms.append(A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=1.0))
                elif i == 6:
                    transforms.append(A.Compose([A.RandomBrightness(limit=0.1, p=1.0), A.Blur(blur_limit=2, p=1.0)], p=1.0))
                elif i == 7:
                    transforms.append(A.Compose([A.RandomBrightness(limit=-0.1, p=1.0), A.Blur(blur_limit=2, p=1.0)], p=1.0))
                elif i == 8:
                    transforms.append(A.Compose([A.RandomBrightness(limit=0.1, p=1.0), A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=1.0)], p=1.0))
                elif i == 9:
                    transforms.append(A.Compose([A.RandomBrightness(limit=-0.1, p=1.0), A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=1.0)], p=1.0))

                bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'])
                if len(transforms) > 0:
                    transform = A.Compose(transforms, bbox_params=bbox_params, p=1.0)
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_ids)
                    augmented_image = augmented['image']
                    augmented_bboxes = augmented['bboxes']

            unique_id = str(uuid.uuid4())[:8]  # Generate a unique identifier for each augmented image
            output_image_path = os.path.join(output_folder, 'test', 'images', f"{os.path.splitext(image_file)[0]}_{unique_id}_{i}.jpg")
            cv2.imwrite(output_image_path, augmented_image)

            output_label_path = os.path.join(output_folder, 'test', 'labels', f"{os.path.splitext(image_file)[0]}_{unique_id}_{i}.txt")
            with open(output_label_path, 'w') as file:
                for class_id, bbox in zip(class_ids, augmented_bboxes):
                    file.write(f"{class_id} {' '.join(map(str, bbox))}\n")
    print("Done")

if __name__ == "__main__":
    custom_dataset_folder = "Hero_name_recognition.v4-notaugmented.yolov5pytorch"
    images_folder = os.path.join(custom_dataset_folder, "test\images")
    labels_folder = os.path.join(custom_dataset_folder, "test\labels")
    augmented_dataset_folder = "augmented_dataset"

    augment_images(images_folder, labels_folder, augmented_dataset_folder)
