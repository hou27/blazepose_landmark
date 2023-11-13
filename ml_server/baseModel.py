import pickle
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier


class BaseModel:
    def __init__(self, model_path, scaler_path):
        with open(model_path, "rb") as f:
            ensemble_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        self.model: VotingClassifier = ensemble_model
        self.scaler: MinMaxScaler = scaler

    def predict(self, keypoint_3d):
        preprocessed_data = self.preprocess_base(keypoint_3d)
        if preprocessed_data is None:
            return None
        return self.model.predict(preprocessed_data)

    def preprocess_base(self, keypoint_3d, score_threshold):
        if self.scaler is None:
            raise ValueError("Scaler not loaded.")
        data = np.array(
            [np.array([d["x"], d["y"], d["z"], d["score"]]) for d in keypoint_3d]
        )

        # score 값이 threshold 이하인 좌표가 10개 이상이면 처리 종료
        if len(data[data[:, 3] < score_threshold]) >= 10:
            print("under score_threshold")
            return None

        # data에서 score값 제거
        data = data[:, :-1]

        # 엉덩이를 기준으로 좌표 재설정
        hip_coords = data[23]
        for i in range(data.shape[0]):
            data[i] -= hip_coords

        return data

    def __calculate_angle(self, a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def __calculate_xy_angle(self, a, b, c):
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]

        dot_product = ba[0] * bc[0] + ba[1] * bc[1]

        magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        cos_theta = dot_product / (magnitude_ba * magnitude_bc)
        angle_in_degree = math.acos(cos_theta) * (180 / math.pi)

        return angle_in_degree
