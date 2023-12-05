import numpy as np

from baseModel import BaseModel


class PushupEnsembleModel(BaseModel):
    def __init__(self):
        super().__init__(
            "./model/pushup_ensemble_model.pkl",
            "./scaler/pushup_scaler1020.pkl",
        )

    def preprocess_base(self, keypoint_3d):
        data = super().preprocess_base(keypoint_3d, 0.1)
        if data is None:
            return None

        print(data)

        # 엉덩이를 기준으로 좌표 재설정
        hip_coords = data[23]
        for i in range(data.shape[0]):
            data[i] -= hip_coords

        # 팔 다리 각도 계산

        # Right
        shoulder_right = data[11]
        elbow_right = data[13]
        wrist_right = data[15]
        hip_right = data[23]
        knee_right = data[25]
        ankle_right = data[27]

        # Left
        shoulder_left = data[12]
        elbow_left = data[14]
        wrist_left = data[16]
        hip_left = data[24]
        knee_left = data[26]
        ankle_left = data[28]

        angle_right_arm = self.calculate_angle(shoulder_right, elbow_right, wrist_right)
        angle_left_arm = self.calculate_angle(shoulder_left, elbow_left, wrist_left)
        angle_right_leg = self.calculate_angle(hip_right, knee_right, ankle_right)
        angle_left_leg = self.calculate_angle(hip_left, knee_left, ankle_left)

        xy_angle_right_arm = self.calculate_xy_angle(
            shoulder_right[:-1], elbow_right[:-1], wrist_right[:-1]
        )
        xy_angle_left_arm = self.calculate_xy_angle(
            shoulder_left[:-1], elbow_left[:-1], wrist_left[:-1]
        )
        xy_angle_right_leg = self.calculate_xy_angle(
            hip_right[:-1], knee_right[:-1], ankle_right[:-1]
        )
        xy_angle_left_leg = self.calculate_xy_angle(
            hip_left[:-1], knee_left[:-1], ankle_left[:-1]
        )

        data = np.append(data, round(angle_right_arm, 2))
        data = np.append(data, round(angle_left_arm, 2))
        data = np.append(data, round(angle_right_leg, 2))
        data = np.append(data, round(angle_left_leg, 2))
        data = np.append(data, round(xy_angle_right_arm, 2))
        data = np.append(data, round(xy_angle_left_arm, 2))
        data = np.append(data, round(xy_angle_right_leg, 2))
        data = np.append(data, round(xy_angle_left_leg, 2))

        normalized_data = self.scaler.transform([data])
        preprocessed_data = normalized_data[0][-8:]

        # 팔 다리 각도 값 조절(가중치 조절)
        preprocessed_data[[0, 1, 4, 5]] = (
            np.square(preprocessed_data[[0, 1, 4, 5]] * 4) / 2
        )
        preprocessed_data[[2, 3, 6, 7]] = preprocessed_data[[2, 3, 6, 7]] * 4

        return [preprocessed_data]  # 2차원 배열로 반환
