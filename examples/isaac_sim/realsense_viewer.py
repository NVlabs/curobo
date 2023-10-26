#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Third Party
import cv2
import numpy as np
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader


def view_realsense():
    realsense_data = RealsenseDataloader(clipping_distance_m=1.0)
    # Streaming loop
    try:
        while True:
            data = realsense_data.get_raw_data()
            depth_image = data[0]
            color_image = data[1]
            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        realsense_data.stop_device()


if __name__ == "__main__":
    view_realsense()
