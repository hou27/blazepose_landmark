# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from draw_landmarks_on_image import draw_landmarks_on_image

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)
image1 = "038-1-1-21-Z17_D-0000004.jpg"
image2 = "506-1-2-23-Z57_A-0000009.jpg"
image3 = "561-1-3-27-Z37_C-0000001.jpg"
# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")
image = mp.Image.create_from_file(image3)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# 아무 키 입력을 기다립니다. 키 입력을 받으면 이미지 창이 닫힙니다.
cv2.waitKey(0)

# 모든 창을 닫습니다.
cv2.destroyAllWindows()
