import numpy as np
import cv2
import os
from typing import Dict, Tuple
import yaml

def load_config(config_path: str = 'configs/params.yaml') -> Dict:
    """Завантаження конфігураційних параметрів"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_raw_data(file_path: str) -> Dict:
    """
    Завантаження сирих даних з .npy файлу
    Повертає словник з:
    - params: параметри зйомки
    - data: 3D масив даних (traces x crosslines x inlines)
    - labels: мітки (0/1)
    """
    data = np.load(file_path, allow_pickle=True).item()
    return {
        'params': data['param'],
        'data': data['data'],
        'labels': data['ground_truth']
    }

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Нормалізація 3D об'єму до [0, 1]"""
    v_min = np.min(volume)
    v_max = np.max(volume)
    return (volume - v_min) / (v_max - v_min + 1e-8)

def enhance_contrast(volume: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Підсилення контрасту за допомогою CLAHE"""
    enhanced = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        slice_8bit = ((volume[i] * 255).astype(np.uint8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced[i] = clahe.apply(slice_8bit) / 255.0
    return enhanced

def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """
    Основна функція попередньої обробки:
    1. Нормалізація
    2. Підсилення контрасту
    3. Фільтрація (медіанний фільтр)
    """
    volume = normalize_volume(volume)
    volume = enhance_contrast(volume)
    
    # Застосування 3D медіанного фільтру
    filtered = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        filtered[i] = cv2.medianBlur((volume[i] * 255).astype(np.uint8), 3) / 255.0
    return filtered

def split_dataset(data: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Tuple:
    """Розділення даних на тренувальні та тестові"""
    idx = int(data.shape[0] * (1 - test_size))
    return (
        data[:idx], labels[:idx],  # train
        data[idx:], labels[idx:]   # test
    )

def save_processed_data(data: np.ndarray, labels: np.ndarray, output_dir: str):
    """Збереження оброблених даних"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'processed_data.npy'), data)
    np.save(os.path.join(output_dir, 'processed_labels.npy'), labels)

def full_preprocessing_pipeline(input_path: str, output_dir: str):
    """Повний пайплайн обробки даних"""
    # Завантаження
    raw_data = load_raw_data(input_path)
    
    # Обробка
    processed_volume = preprocess_volume(raw_data['data'])
    
    # Розділення
    X_train, y_train, X_test, y_test = split_dataset(processed_volume, raw_data['labels'])
    
    # Збереження
    save_processed_data(X_train, y_train, os.path.join(output_dir, 'train'))
    save_processed_data(X_test, y_test, os.path.join(output_dir, 'test'))
    
    print(f"Обробка завершена. Дані збережено в {output_dir}")

if __name__ == "__main__":
    config = load_config()
    input_file = os.path.join('data', 'raw', config['data']['input_file'])
    output_dir = os.path.join('data', 'processed')
    
    full_preprocessing_pipeline(input_file, output_dir) 