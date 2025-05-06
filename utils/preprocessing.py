import numpy as np
import cv2
import os
from typing import Dict, Tuple
import yaml

def load_config(config_path: str = 'configs/params.yaml') -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_raw_data(file_path: str) -> Dict:
    data = np.load(file_path, allow_pickle=True).item()
    return {
        'params': data['param'],
        'data': data['data'],
        'labels': data['ground_truth']
    }

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    v_min = np.min(volume)
    v_max = np.max(volume)
    return (volume - v_min) / (v_max - v_min + 1e-8)

def preprocess_volume(volume: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    processed = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        slice_8bit = (volume[i] * 255).astype(np.uint8)
        processed[i] = cv2.createCLAHE(clipLimit=clip_limit).apply(slice_8bit) / 255.0
    return processed

def save_processed_data(data: np.ndarray, labels: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'processed_data.npy'), data)
    np.save(os.path.join(output_dir, 'processed_labels.npy'), labels)

def load_processed_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(os.path.join(data_dir, 'processed_data.npy'))
    labels = np.load(os.path.join(data_dir, 'processed_labels.npy'))
    return data, labels

def full_preprocessing_pipeline(input_path: str, output_dir: str):
    raw = load_raw_data(input_path)
    norm_data = normalize_volume(raw['data'])
    proc_data = preprocess_volume(norm_data)
    save_processed_data(proc_data, raw['labels'], output_dir)

def prepare_3d_input(data: np.ndarray) -> np.ndarray:
    """Convert data to 5D tensor (batch, depth, height, width, channels)"""
    return np.expand_dims(data, axis=-1)