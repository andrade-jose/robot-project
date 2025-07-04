import tensorflow as tf
import numpy as np
from config.config_advanced import config

def pose_loss(y_true, y_pred):
    # Função de perda para pose 3D
    t_true, r_true = y_true[:, :3], y_true[:, 3:]
    t_pred, r_pred = y_pred[:, :3], y_pred[:, 3:]
    # Normaliza os vetores de rotação
    loss_t = tf.reduce_mean(tf.square(t_true - t_pred))
    
    dot = tf.reduce_sum(r_true * r_pred, axis=1)
    dot = tf.clip_by_value(dot, -1.0, 1.0)  # evita domínio inválido para acos
    loss_r = tf.reduce_mean(tf.acos(2 * tf.square(dot) - 1))

    return config.POSE_LOSS_WEIGHTS[0] * loss_t + config.POSE_LOSS_WEIGHTS[1] * loss_r


def translation_error(y_true, y_pred):
    """RMSE para translação (em unidades normalizadas)"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, :3] - y_pred[:, :3])))

def rotation_error(y_true, y_pred):
    """Erro angular médio em graus"""
    dot = tf.reduce_sum(y_true[:, 3:] * y_pred[:, 3:], axis=1)
    return tf.reduce_mean(tf.acos(2 * tf.square(dot) - 1) * (180 / np.pi))