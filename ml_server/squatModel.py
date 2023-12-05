import numpy as np

from baseModel import BaseModel

# path = "/Users/hou27/workspace/ml/blazepose_landmark/ml_server"


class SquatEnsembleModel(BaseModel):
    def __init__(self):
        super().__init__(
            "./model/squat_ensemble_model.pkl",
            "./scaler/squat_scaler1020.pkl",
        )

    def preprocess_base(self, keypoint_3d):
        data = super().preprocess_base(keypoint_3d, 0.001)
        if data is None:
            return None

        # 엉덩이를 기준으로 좌표 재설정
        hip_coords = data[23]
        for i in range(data.shape[0]):
            data[i] -= hip_coords

        # 팔 다리 각도 계산

        # Right
        hip_right = data[23]
        knee_right = data[25]
        ankle_right = data[27]

        # Left
        hip_left = data[24]
        knee_left = data[26]
        ankle_left = data[28]

        angle_right_leg = self.__calculate_angle(hip_right, knee_right, ankle_right)
        angle_left_leg = self.__calculate_angle(hip_left, knee_left, ankle_left)

        xy_angle_right_leg = self.__calculate_xy_angle(
            hip_right[:-1], knee_right[:-1], ankle_right[:-1]
        )
        xy_angle_left_leg = self.__calculate_xy_angle(
            hip_left[:-1], knee_left[:-1], ankle_left[:-1]
        )

        data = np.append(data, round(angle_right_leg, 2))
        data = np.append(data, round(angle_left_leg, 2))
        data = np.append(data, round(xy_angle_right_leg, 2))
        data = np.append(data, round(xy_angle_left_leg, 2))

        normalized_data = self.scaler.transform([data])
        preprocessed_data = normalized_data[0][-4:]

        # 팔 다리 각도 값 조절(가중치 조절)
        preprocessed_data = preprocessed_data * 2

        return [preprocessed_data]  # 2차원 배열로 반환


# class SquatEnsembleModel:
#     def __init__(self):
#         with open(path + "/model/squart_ensemble_model.pkl", "rb") as f:
#             ensemble_model = pickle.load(f)
#         with open(path + "/scaler/squart_scaler1004.pkl", "rb") as f:
#             scaler = pickle.load(f)
#         self.model: VotingClassifier = ensemble_model
#         self.scaler: MinMaxScaler = scaler

#     def predict(self, keypoint_3d):
#         preprocessed_data = self.__preprocess(keypoint_3d)
#         if preprocessed_data is None:
#             return None
#         return self.model.predict(preprocessed_data)

#     def __preprocess(self, keypoint_3d):
#         if self.scaler is None:
#             raise ValueError("Scaler not loaded.")
#         data = np.array(
#             [np.array([d["x"], d["y"], d["z"], d["score"]]) for d in keypoint_3d]
#         )

#         # score 값이 0.01 이하인 좌표가 10개 이상이면 처리 종료
#         if len(data[data[:, 3] < 0.01]) >= 10:
#             return None

#         # data에서 score값 제거
#         data = data[:, :-1]

# # 엉덩이를 기준으로 좌표 재설정
# hip_coords = data[23]
# for i in range(data.shape[0]):
#     data[i] -= hip_coords

# # 팔 다리 각도 계산

# # Right
# hip_right = data[23]
# knee_right = data[25]
# ankle_right = data[27]

# # Left
# hip_left = data[24]
# knee_left = data[26]
# ankle_left = data[28]

# angle_right_leg = self.__calculate_angle(hip_right, knee_right, ankle_right)
# angle_left_leg = self.__calculate_angle(hip_left, knee_left, ankle_left)

# xy_angle_right_leg = self.__calculate_xy_angle(
#     hip_right[:-1], knee_right[:-1], ankle_right[:-1]
# )
# xy_angle_left_leg = self.__calculate_xy_angle(
#     hip_left[:-1], knee_left[:-1], ankle_left[:-1]
# )

# data = np.append(data, round(angle_right_leg, 2))
# data = np.append(data, round(angle_left_leg, 2))
# data = np.append(data, round(xy_angle_right_leg, 2))
# data = np.append(data, round(xy_angle_left_leg, 2))

# normalized_data = self.scaler.transform([data])
# preprocessed_data = normalized_data[0][-4:]

# # 팔 다리 각도 값 조절(가중치 조절)
# preprocessed_data = preprocessed_data * 2

# return [preprocessed_data]  # 2차원 배열로 반환

#     def __calculate_angle(self, a, b, c):
#         ba = a - b  # vector from point b to a
#         bc = c - b  # vector from point b to c

#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         angle = np.arccos(cosine_angle)

#         return np.degrees(angle)

#     def __calculate_xy_angle(self, a, b, c):
#         # 벡터 생성
#         ba = [a[0] - b[0], a[1] - b[1]]  # 벡터 BA
#         bc = [c[0] - b[0], c[1] - b[1]]  # 벡터 BC

#         # 내적 계산
#         dot_product = ba[0] * bc[0] + ba[1] * bc[1]

#         # 두 벡터의 크기 계산
#         magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
#         magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

#         # cos(theta) 계산
#         cos_theta = dot_product / (magnitude_ba * magnitude_bc)

#         # acos(cos_theta)를 사용하여 theta(라디안 단위) 찾기, 그리고 degree로 변환
#         angle_in_degree = math.acos(cos_theta) * (180 / math.pi)

#         return angle_in_degree
