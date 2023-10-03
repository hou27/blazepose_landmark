# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

from draw_landmarks_on_image import draw_landmarks_on_image

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path="./model/pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# list for keypoint data
landmarks_data = []

# load img files
image_directory1 = "./image/561-1-3-27-Z37_C"
image_directory2 = "./image/561-1-3-27-Z54_C"
image_directory3 = "./image/561-1-3-27-Z56_C"
image_directory4 = "./image/561-1-3-27-Z115_C"
image_directory5 = "./image/561-1-3-27-Z59_C"
image_directory6 = "./image/561-1-3-27-Z62_C"
image_directory7 = "./image/561-1-3-27-Z61_C"
image_directory8 = "./image/561-1-3-27-Z84_C"
image_directory9 = "./image/561-1-3-27-Z3_C"
image_directory10 = "./image/561-1-3-27-Z98_C"
image_directorys = [
    image_directory1,
    image_directory2,
    image_directory3,
    image_directory4,
    image_directory5,
    image_directory6,
    image_directory7,
    image_directory8,
    image_directory9,
    image_directory10,
]
# image_directory = image_directory10
# image_files = [f for f in os.listdir(image_directory) if f.endswith((".jpg", ".jpeg"))]

# detect pose landmarks from the input image and save landmarks to npy file
for image_directory in image_directorys:
    image_files = [
        f for f in os.listdir(image_directory) if f.endswith((".jpg", ".jpeg"))
    ]
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = mp.Image.create_from_file(image_path)

        # Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        # Process the detection result.
        landmarks = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in detection_result.pose_landmarks[0]
        ]

        # save landmarks to npy file
        # np.save(f"./npy/train/{image_file}.npy", np.array(landmarks))
        np.save(
            f"./npy/train_entire/{image_file}.npy", np.array(landmarks)
        )  # add to test folder

# image1 = "038-1-1-21-Z17_D-0000004.jpg"
# image2 = "506-1-2-23-Z57_A-0000009.jpg"
# image3 = "561-1-3-27-Z37_C-0000001.jpg"
# image4 = "test_converted.jpg"
# image5 = "stand_body_converted.jpg"
# image6 = "mingyu_stand_converted.jpg"
# image7 = "561-1-3-27-Z37_C-0000009.jpg"
# image8 = "561-1-3-27-Z56_C-0000001.jpg"
# image9 = "561-1-3-27-Z56_C-0000004.jpg"
# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")
# image = mp.Image.create_from_file(image9)  # jpg만 가능

# STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)

# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# print(detection_result.pose_landmarks)

"""
# step 5부터 주석 처리 및 landmark 출력 코드로 변경

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# 아무 키 입력을 기다립니다. 키 입력을 받으면 이미지 창이 닫힙니다.
cv2.waitKey(0)

# 모든 창을 닫습니다.
cv2.destroyAllWindows()
"""
