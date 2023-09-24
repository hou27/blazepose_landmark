import sys
import pickle
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier

path = "/Users/hou27/workspace/ml/blazepose_landmark/blazepose_web/src/python"


class EnsembleModel:
    def __init__(self):
        with open(path + "/model/ensemble_model.pkl", "rb") as f:
            ensemble_model = pickle.load(f)
        with open(path + "/scaler/scaler0924.pkl", "rb") as f:
            scaler = pickle.load(f)
        self.model: VotingClassifier = ensemble_model
        self.scaler: MinMaxScaler = scaler

    def predict(self, keypoint_3d):
        preprocessed_data = self.__preprocess(keypoint_3d)
        return self.model.predict(preprocessed_data)

    def __preprocess(self, keypoint_3d):
        if self.scaler is None:
            raise ValueError("Scaler not loaded.")
        data = np.array([np.array([d["x"], d["y"], d["z"]]) for d in keypoint_3d])

        # 엉덩이를 기준으로 좌표 재설정
        hip_coords = data[23]
        for i in range(data.shape[0]):
            data[i] -= hip_coords

        # 팔 각도 계산
        shoulder_right = data[11]
        elbow_right = data[13]
        wrist_right = data[15]

        shoulder_left = data[12]
        elbow_left = data[14]
        wrist_left = data[16]

        angle_right_arm = self.__calculate_angle(
            shoulder_right, elbow_right, wrist_right
        )
        angle_left_arm = self.__calculate_angle(shoulder_left, elbow_left, wrist_left)

        xy_angle_right_arm = self.__calculate_xy_angle(
            shoulder_right[:-1], elbow_right[:-1], wrist_right[:-1]
        )
        xy_angle_left_arm = self.__calculate_xy_angle(
            shoulder_left[:-1], elbow_left[:-1], wrist_left[:-1]
        )

        data = np.append(data, round(angle_right_arm, 2))
        data = np.append(data, round(angle_left_arm, 2))
        data = np.append(data, round(xy_angle_right_arm, 2))
        data = np.append(data, round(xy_angle_left_arm, 2))

        normalized_data = self.scaler.transform([data])
        preprocessed_data = normalized_data[0][-4:]

        return [preprocessed_data]  # 2차원 배열로 반환

    def __calculate_angle(self, a, b, c):
        ba = a - b  # vector from point b to a
        bc = c - b  # vector from point b to c

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def __calculate_xy_angle(self, a, b, c):
        # 벡터 생성
        ba = [a[0] - b[0], a[1] - b[1]]  # 벡터 BA
        bc = [c[0] - b[0], c[1] - b[1]]  # 벡터 BC

        # 내적 계산
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]

        # 두 벡터의 크기 계산
        magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        # cos(theta) 계산
        cos_theta = dot_product / (magnitude_ba * magnitude_bc)

        # acos(cos_theta)를 사용하여 theta(라디안 단위) 찾기, 그리고 degree로 변환
        angle_in_degree = math.acos(cos_theta) * (180 / math.pi)

        return angle_in_degree


if __name__ == "__main__":
    model = EnsembleModel()
    print(sys.argv[1])
    data = sys.argv[1]
    print(model.predict(data))
