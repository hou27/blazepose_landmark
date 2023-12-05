import numpy as np

from baseModel import BaseModel


class SquatEnsembleModel(BaseModel):
    def __init__(self):
        super().__init__(
            "./model/squat_ensemble_model.pkl",
            "./scaler/squat_scaler1020.pkl",
        )

    def preprocess_base(self, keypoint_3d):
        data = super().preprocess_base(keypoint_3d, 0.1)
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

        angle_right_leg = self.calculate_angle(hip_right, knee_right, ankle_right)
        angle_left_leg = self.calculate_angle(hip_left, knee_left, ankle_left)

        xy_angle_right_leg = self.calculate_xy_angle(
            hip_right[:-1], knee_right[:-1], ankle_right[:-1]
        )
        xy_angle_left_leg = self.calculate_xy_angle(
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
