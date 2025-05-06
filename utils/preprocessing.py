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
        'data': data['data'].astype(np.float32),
        'labels': data['ground_truth'].astype(np.int32)
    }

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    return (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)

def preprocess_volume(volume: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Process volume without adding extra dimensions"""
    processed = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        slice_8bit = (volume[i] * 255).astype(np.uint8)
        processed[i] = cv2.createCLAHE(clipLimit=clip_limit).apply(slice_8bit) / 255.0
    return processed

def save_processed(data: np.ndarray, labels: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'data.npy'), data)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

def full_pipeline(input_path: str, output_dir: str):
    """Produces properly shaped (N,66,170,440,1) output"""
    config = load_config()
    raw = load_raw_data(input_path)
    
    # Process without adding dimensions
    norm_data = normalize_volume(raw['data'])
    proc_data = preprocess_volume(norm_data, config['preprocessing']['clip_limit'])
    
    # Only add channel dimension if missing
    if proc_data.ndim == 3:
        proc_data = np.expand_dims(proc_data, axis=-1)
    
    save_processed(proc_data, raw['labels'], output_dir)

def load_processed_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load processed data and ensure proper 5D tensor shape for Conv3D
    Returns:
        data: (1, depth, height, width, channels) float32
        labels: (depth,) int32
    """
    # Load numpy arrays
    data = np.load(os.path.join(data_dir, 'data.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    
    # Ensure proper shape (d,h,w,c)
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)  # Add channel if missing
    
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Add batch dimension (1,d,h,w,c)
    data = np.expand_dims(data, axis=0)
    
    return data, labels
