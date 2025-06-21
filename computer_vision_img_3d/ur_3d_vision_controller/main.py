from system.vision_to_motion_controller import VisionToMotionController

CONFIG = {
    'camera': {
        'width': 640,
        'height': 480,
        'fps': 30
    },
    'model_path': 'models/object_detector.h5',
    'calibration_file': 'config/calibration.json',
    'robot_ip': '192.168.1.10'
}

def main():
    system = VisionToMotionController(
        camera_config=CONFIG['camera'],
        model_path=CONFIG['model_path'],
        calibration_file=CONFIG['calibration_file'],
        robot_ip=CONFIG['robot_ip']
    )
    try:
        while True:
            success = system.run_single_cycle()
            print(f"Ciclo {'bem-sucedido' if success else 'sem detecção'}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Parando sistema...")
    finally:
        system.shutdown()

if __name__ == '__main__':
    main()
