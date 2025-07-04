# stereo_vision/utils.py

import os
import cv2
import numpy as np

def draw_horizontal_lines(img, line_color=(0, 255, 0), thickness=1, spacing=40):
    """
    Desenha linhas horizontais na imagem para verificar o alinhamento.

    Args:
        img (np.ndarray): imagem BGR.
        line_color (tuple): cor da linha (default: verde).
        thickness (int): espessura da linha.
        spacing (int): espaçamento entre linhas.

    Returns:
        np.ndarray: imagem com linhas desenhadas.
    """
    img_lines = img.copy()
    h, w = img.shape[:2]
    for y in range(spacing, h, spacing):
        cv2.line(img_lines, (0, y), (w, y), line_color, thickness)
    return img_lines

def show_side_by_side_with_lines(img_left, img_right, spacing=40):
    """
    Mostra imagens lado a lado com linhas horizontais desenhadas.

    Args:
        img_left (np.ndarray): imagem esquerda.
        img_right (np.ndarray): imagem direita.
        spacing (int): espaçamento entre linhas.
    """
    img_left_lines = draw_horizontal_lines(img_left, spacing=spacing)
    img_right_lines = draw_horizontal_lines(img_right, spacing=spacing)
    combined = np.hstack((img_left_lines, img_right_lines))
    cv2.imshow('Retificação - Linhas Horizontais', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def list_image_pairs(left_dir, right_dir):
    """
    Retorna listas ordenadas de imagens pareadas (assume nomes compatíveis).

    Args:
        left_dir (str): pasta com imagens esquerda.
        right_dir (str): pasta com imagens direita.

    Returns:
        list: pares (caminho_esq, caminho_dir)
    """
    left_imgs = sorted([f for f in os.listdir(left_dir) if f.lower().endswith(('.png', '.jpg'))])
    right_imgs = sorted([f for f in os.listdir(right_dir) if f.lower().endswith(('.png', '.jpg'))])

    if len(left_imgs) != len(right_imgs):
        raise ValueError("Número diferente de imagens entre pastas esquerda e direita.")

    return [
        (os.path.join(left_dir, l), os.path.join(right_dir, r))
        for l, r in zip(left_imgs, right_imgs)
    ]
