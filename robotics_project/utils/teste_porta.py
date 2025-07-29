from services.robot_service import RobotService, RobotPose

robot = RobotService(".0.0.1")  # ou IP da VM se estiver fora
if robot.connect():
    pose = robot.get_predefined_pose("center")
    if pose:
        robot.move_to_pose(pose)
    robot.disconnect()