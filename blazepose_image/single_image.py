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

image_directory = "./image/561-1-3-27-Z115_C"
image_file = "561-1-3-27-Z115_C-0000032.jpg"
image_path = os.path.join(image_directory, image_file)
image = mp.Image.create_from_file(image_path)

# Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# Process the detection result.
landmarks = [
    (landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence)
    for landmark in detection_result.pose_landmarks[0]
]

# save landmarks to npy file
# np.save(f"./npy/train/{image_file}.npy", np.array(landmarks))
np.save(f"./{image_file}.npy", np.array(landmarks))  # add to test folder

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

# step 5부터 주석 처리 및 landmark 출력 코드로 변경

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# 아무 키 입력을 기다립니다. 키 입력을 받으면 이미지 창이 닫힙니다.
cv2.waitKey(0)

# 모든 창을 닫습니다.
cv2.destroyAllWindows()
