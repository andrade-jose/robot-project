import numpy as np
import json

class CoordinateTransformer:
    def __init__(self, calibration_file):
        with open(calibration_file, 'r') as f:
            data = json.load(f)
        self.matrix = np.array(data['transformation_matrix'])
        self.offset = np.array(data.get('tool_offset', [0, 0, 0]))
    
    def camera_to_robot(self, camera_coords):
        homog = np.append(camera_coords, 1)
        robot_coords = self.matrix @ homog
        return robot_coords[:3] + self.offset
    
    def get_orientation(self, height):
        return (np.pi, 0, 0) if height > 0.3 else (0, np.pi/2, 0)