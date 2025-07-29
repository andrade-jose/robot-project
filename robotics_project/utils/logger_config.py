# NOVO ARQUIVO: utils/logger_config.py
import logging
import sys
from datetime import datetime

def configurar_logger():
    """Configurar sistema de logging estruturado"""
    
    # Formatter personalizado
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(
        f'logs/tapatan_robot_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)
    
    # Logger principal
    logger = logging.getLogger('TapatanRobot')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
