# scripts/live_demo.py

import cv2
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stereo_vision.capture import StereoCameras
from stereo_vision.processor import StereoProcessor
from config.config import CALIB_FILE

def main():
    print("[INFO] Carregando calibração...")
    processor = StereoProcessor(str(CALIB_FILE))

    print("[INFO] Iniciando câmeras estéreo...")
    cams = StereoCameras(left_cam_id=0, right_cam_id=1)
    cams.set_rectification_maps(
        processor.left_map1, processor.left_map2,
        processor.right_map1, processor.right_map2
    )

    while True:
        try:
            frame_left, frame_right = cams.get_frames(rectify=True)

            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            disparity = processor.disparity_calc.compute(gray_left, gray_right)
            disp_vis = processor.disparity_calc.normalize(disparity)

            depth_map = processor.reconstructor.get_depth_map(
                processor.reconstructor.compute_point_cloud(disparity)
            )
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

            cv2.imshow("Disparidade | Profundidade", cv2.hconcat([disp_vis, depth_vis]))

            if cv2.waitKey(1) == 27:  # ESC
                break

        except Exception as e:
            print(f"[ERRO] {e}")
            break

    cams.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
