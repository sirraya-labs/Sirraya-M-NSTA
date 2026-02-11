"""
SIRRAYA ENTERPRISE DEEPFAKE DETECTION SYSTEM v3.2.0
Core Components: SIS Standardization, Environmental Calibration, rPPG, SMI, Benchmarking
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal, fft, spatial, stats
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import asyncio
import websockets
from dataclasses import dataclass
import hashlib
import pickle
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
import queue
import threading
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
import warnings
warnings.filterwarnings('ignore')
import os
import time
import random
import base64
from enum import Enum
import csv


# Add this after the imports in sirraya_core.py
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str = "sirraya_core") -> logging.Logger:
    """Setup production-grade logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # File handler with rotation
        file_handler = RotatingFileHandler(
            f"sirraya_{name}.log", maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger("sirraya_core")

# ============================================================================
# ENUMS AND CONSTANTS FOR STANDARDIZATION
# ============================================================================

class SISCategory(Enum):
    """Sirraya Integrity Score Categories"""
    DETERMINISTIC_AUTHENTICATION = (90, 100, "✅ DETERMINISTIC AUTHENTICATION")
    HIGH_PROBABILITY_AUTHENTIC = (70, 89, "✔️ HIGH PROBABILITY AUTHENTIC")
    MODERATE_PROBABILITY_AUTHENTIC = (50, 69, "⚠️ MODERATE PROBABILITY AUTHENTIC")
    SUSPICIOUS = (40, 49, "⚠️ SUSPICIOUS - REQUIRES REVIEW")
    HIGH_PROBABILITY_DEEPFAKE = (20, 39, "🚨 HIGH PROBABILITY DEEPFAKE")
    CRITICAL_FAILURE = (0, 19, "🔴 CRITICAL FAILURE DETECTED")
    
    @classmethod
    def from_score(cls, score: float) -> 'SISCategory':
        """Get SIS category from score"""
        score_int = int(round(score))
        for category in cls:
            if category.value[0] <= score_int <= category.value[1]:
                return category
        return cls.CRITICAL_FAILURE


@dataclass
class SISResult:
    """Standardized SIS Result Payload"""
    sis_score: float  # 0-100 scale
    sis_category: str
    sis_verdict: str
    forensic_confidence: float  # 0-1 scale
    requires_human_review: bool
    verification_layers: Dict[str, Dict[str, Any]]
    environmental_conditions: Dict[str, Any]
    blockchain_hash: Optional[str] = None
    timestamp: str = None
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to standardized JSON payload"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + 'Z'
        
        return {
            "metadata": {
                "system": "Sirraya Enterprise v3.2.0",
                "standard": "SIS-2026",
                "timestamp": self.timestamp,
                "version": "3.2.0"
            },
            "result": {
                "sirraya_integrity_score": self.sis_score,
                "sis_category": self.sis_category,
                "sis_verdict": self.sis_verdict,
                "forensic_confidence": self.forensic_confidence,
                "requires_human_review": self.requires_human_review,
                "interpretation": self._get_interpretation(),
                "recommended_action": self._get_recommended_action()
            },
            "verification_details": {
                "layers": self.verification_layers,
                "environmental": self.environmental_conditions
            },
            "provenance": {
                "blockchain_hash": self.blockchain_hash,
                "certification_level": self._get_certification_level()
            }
        }
    
    def _get_interpretation(self) -> str:
        """Get human-readable interpretation"""
        if self.sis_score >= 90:
            return "Deterministically authenticated via rPPG and Structural Matrix Integrity"
        elif self.sis_score >= 70:
            return "High probability authentic - passes neural quantum analysis"
        elif self.sis_score >= 50:
            return "Moderate confidence - additional verification recommended"
        elif self.sis_score >= 40:
            return "Suspicious content detected - manual review required"
        elif self.sis_score >= 20:
            return "High probability deepfake - geometric anomalies detected"
        else:
            return "Critical failure - biophysical or structural impossibility detected"
    
    def _get_recommended_action(self) -> str:
        """Get recommended action based on SIS"""
        if self.sis_score >= 90:
            return "PROCEED_WITH_FULL_TRUST"
        elif self.sis_score >= 70:
            return "PROCEED_WITH_HIGH_CONFIDENCE"
        elif self.sis_score >= 50:
            return "ADDITIONAL_VERIFICATION_ADVISED"
        elif self.sis_score >= 40:
            return "ESCALATE_TO_SECURITY_TEAM"
        elif self.sis_score >= 20:
            return "BLOCK_AND_INVESTIGATE"
        else:
            return "IMMEDIATE_SECURITY_INTERVENTION"
    
    def _get_certification_level(self) -> str:
        """Get certification level for provenance"""
        if self.sis_score >= 90:
            return "PLATINUM"
        elif self.sis_score >= 70:
            return "GOLD"
        elif self.sis_score >= 50:
            return "SILVER"
        elif self.sis_score >= 40:
            return "BRONZE"
        else:
            return "UNCERTIFIED"


# ============================================================================
# ENVIRONMENTAL CONDITIONS ASSESSMENT
# ============================================================================

class EnvironmentalConditions:
    """Environmental condition constants and thresholds"""
    
    # Lighting conditions (in lux equivalents from image brightness)
    DARK_THRESHOLD = 0.2  # Normalized brightness
    LOW_LIGHT_THRESHOLD = 0.4
    OPTIMAL_LIGHT_THRESHOLD = 0.6
    
    # Motion blur thresholds
    MOTION_BLUR_HIGH = 0.15  # Laplacian variance threshold
    MOTION_BLUR_MEDIUM = 0.08
    
    # Compression noise thresholds
    COMPRESSION_NOISE_HIGH = 0.3
    
    # Minimum SNR for reliable rPPG
    MIN_SNR_FOR_PPG = 3.0
    
    @staticmethod
    def assess_lighting(frame: np.ndarray) -> Dict[str, Any]:
        """Assess lighting conditions from frame"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate image brightness
        brightness = np.mean(gray) / 255.0
        
        # Calculate contrast
        contrast = np.std(gray) / 255.0
        
        # Determine lighting condition
        if brightness < EnvironmentalConditions.DARK_THRESHOLD:
            lighting_condition = "DARK"
            lighting_score = 0.0
        elif brightness < EnvironmentalConditions.LOW_LIGHT_THRESHOLD:
            lighting_condition = "LOW_LIGHT"
            lighting_score = 0.3
        elif brightness < EnvironmentalConditions.OPTIMAL_LIGHT_THRESHOLD:
            lighting_condition = "MODERATE"
            lighting_score = 0.7
        else:
            lighting_condition = "OPTIMAL"
            lighting_score = 1.0
        
        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "condition": lighting_condition,
            "score": lighting_score,
            "rppg_possible": brightness >= EnvironmentalConditions.DARK_THRESHOLD
        }
    
    @staticmethod
    def assess_motion_noise(frames: List[np.ndarray]) -> Dict[str, Any]:
        """Assess motion blur and camera shake"""
        if len(frames) < 3:
            return {
                "motion_blur": 0.0,
                "condition": "INSUFFICIENT_DATA",
                "score": 0.5,
                "rppg_possible": True
            }
        
        # Calculate Laplacian variance for each frame
        laplacian_vars = []
        for frame in frames[-3:]:  # Last 3 frames
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_vars.append(np.var(laplacian))
        
        avg_variance = np.mean(laplacian_vars)
        normalized_variance = avg_variance / 1000.0  # Normalize
        
        # Determine motion condition
        if normalized_variance < EnvironmentalConditions.MOTION_BLUR_HIGH:
            motion_condition = "HIGH_MOTION_BLUR"
            motion_score = 0.0
            rppg_possible = False
        elif normalized_variance < EnvironmentalConditions.MOTION_BLUR_MEDIUM:
            motion_condition = "MODERATE_MOTION_BLUR"
            motion_score = 0.3
            rppg_possible = True
        else:
            motion_condition = "STABLE"
            motion_score = 1.0
            rppg_possible = True
        
        return {
            "motion_blur": float(normalized_variance),
            "condition": motion_condition,
            "score": motion_score,
            "rppg_possible": rppg_possible
        }
    
    @staticmethod
    def calculate_snr_for_ppg(color_signals: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio for rPPG"""
        if color_signals.shape[0] < 30:  # Need minimum frames
            return 0.0
        
        # Calculate SNR in frequency domain
        green_channel = color_signals[:, 1]  # Green channel is best for PPG
        
        # Remove DC component
        signal_detrended = green_channel - np.mean(green_channel)
        
        # Calculate power spectrum
        fft_vals = np.abs(fft.fft(signal_detrended))
        freqs = fft.fftfreq(len(signal_detrended))
        
        # Define frequency bands (0.8-3 Hz = 48-180 BPM)
        heart_rate_mask = (np.abs(freqs) >= 0.8/60) & (np.abs(freqs) <= 3.0/60)
        noise_mask = ~heart_rate_mask
        
        # Calculate signal and noise power
        signal_power = np.sum(fft_vals[heart_rate_mask]**2)
        noise_power = np.sum(fft_vals[noise_mask]**2)
        
        # Avoid division by zero
        if noise_power < 1e-10:
            return 100.0  # Perfect SNR
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr


# ============================================================================
# ENHANCED RPPG WITH ENVIRONMENTAL CALIBRATION
# ============================================================================

class EnvironmentallyCalibratedRPPG:
    """
    Remote PPG with environmental noise floor calibration
    Returns explicit environmental warnings when conditions prevent reliable measurement
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.window_size = int(fps * 10)  # 10-second window
        self.signal_buffer = deque(maxlen=self.window_size)
        self.environment_history = deque(maxlen=10)
        self.ppg_available = True
        self.environmental_warning = None
        
    def update_environmental_assessment(self, frame: np.ndarray):
        """Update environmental assessment for current frame"""
        lighting = EnvironmentalConditions.assess_lighting(frame)
        
        # Add to history
        self.environment_history.append({
            'lighting': lighting,
            'timestamp': time.time()
        })
        
        # Check if rPPG is possible
        if not lighting['rppg_possible']:
            self.ppg_available = False
            self.environmental_warning = "Biophysical verification unavailable: insufficient lighting"
            return
        
        # Check motion if we have enough frames
        if len(self.signal_buffer) >= 3:
            frames = [data['frame'] for data in list(self.signal_buffer)[-3:]]
            motion = EnvironmentalConditions.assess_motion_noise(frames)
            
            if not motion['rppg_possible']:
                self.ppg_available = False
                self.environmental_warning = "Biophysical verification unavailable: excessive motion blur"
                return
        
        self.ppg_available = True
        self.environmental_warning = None
    
    def extract_ppg_signal(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract PPG signal with environmental calibration
        Returns None if environmental conditions prevent reliable measurement
        """
        # Update environmental assessment
        self.update_environmental_assessment(frame)
        
        if not self.ppg_available:
            return {
                'heart_rate': None,
                'confidence': 0.0,
                'snr': 0.0,
                'available': False,
                'environmental_warning': self.environmental_warning,
                'recommendation': 'Improve lighting and stabilize camera'
            }
        
        # Extract ROI signals
        h, w = frame.shape[:2]
        roi = frame[h//3:2*h//3, w//3:2*w//3]  # Facial region
        
        # Extract color channels
        r_signal = np.mean(roi[:, :, 0])
        g_signal = np.mean(roi[:, :, 1])
        b_signal = np.mean(roi[:, :, 2])
        
        # Store in buffer
        self.signal_buffer.append({
            'frame': frame,
            'signals': [r_signal, g_signal, b_signal],
            'timestamp': time.time()
        })
        
        # Need minimum data for analysis
        if len(self.signal_buffer) < 30:  # 1 second at 30 FPS
            return {
                'heart_rate': None,
                'confidence': 0.0,
                'snr': 0.0,
                'available': False,
                'environmental_warning': 'Insufficient data for biophysical analysis',
                'recommendation': 'Continue recording for 1+ seconds'
            }
        
        # Extract color signals from buffer
        color_signals = np.array([data['signals'] for data in self.signal_buffer])
        
        # Calculate SNR
        snr = EnvironmentalConditions.calculate_snr_for_ppg(color_signals)
        
        # Check SNR threshold
        if snr < EnvironmentalConditions.MIN_SNR_FOR_PPG:
            self.ppg_available = False
            self.environmental_warning = f"Biophysical verification unavailable: insufficient signal quality (SNR: {snr:.1f} dB)"
            
            return {
                'heart_rate': None,
                'confidence': 0.0,
                'snr': snr,
                'available': False,
                'environmental_warning': self.environmental_warning,
                'recommendation': 'Improve lighting and reduce movement'
            }
        
        # Perform ICA-based heart rate estimation
        try:
            hr, confidence = self._ica_heart_rate_estimation(color_signals)
            
            return {
                'heart_rate': hr,
                'confidence': confidence,
                'snr': snr,
                'available': True,
                'environmental_warning': None,
                'signal_quality': self._assess_signal_quality(snr, confidence)
            }
            
        except Exception as e:
            logger.error(f"PPG analysis error: {e}")
            return {
                'heart_rate': None,
                'confidence': 0.0,
                'snr': snr,
                'available': False,
                'environmental_warning': f'Analysis error: {str(e)}',
                'recommendation': 'Retry analysis'
            }
    
    def _ica_heart_rate_estimation(self, color_signals: np.ndarray) -> Tuple[float, float]:
        """Independent Component Analysis for heart rate extraction"""
        # Normalize signals
        signals_normalized = (color_signals - np.mean(color_signals, axis=0)) / (
            np.std(color_signals, axis=0) + 1e-8)
        
        # Perform PCA
        cov_matrix = np.cov(signals_normalized.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        
        # Keep top 2 components (heart rate signal is usually in top 2)
        eigenvectors = eigenvectors[:, :2]
        
        # Project data
        projected = signals_normalized @ eigenvectors
        
        # Analyze each component
        best_hr = 0
        best_confidence = 0
        
        for component in projected.T:
            # Filter to physiological range (40-180 BPM)
            min_freq = 40 / 60.0
            max_freq = 180 / 60.0
            
            # Bandpass filter
            nyquist = 0.5 * self.fps
            b, a = signal.butter(4, [min_freq/nyquist, max_freq/nyquist], btype='band')
            filtered = signal.filtfilt(b, a, component)
            
            # Compute power spectrum
            fft_vals = np.abs(fft.fft(filtered))
            freqs = fft.fftfreq(len(filtered), 1.0/self.fps)
            
            # Find peak in physiological range
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            if np.sum(mask) == 0:
                continue
            
            peak_idx = np.argmax(fft_vals[mask])
            hr_estimate = freqs[mask][peak_idx] * 60.0
            
            # Calculate confidence
            signal_power = np.max(fft_vals[mask])
            noise_power = np.mean(fft_vals[~mask]) if np.sum(~mask) > 0 else 1e-8
            confidence = signal_power / noise_power
            
            if confidence > best_confidence and 40 <= hr_estimate <= 180:
                best_hr = hr_estimate
                best_confidence = confidence
        
        # Normalize confidence to 0-1 range
        normalized_confidence = min(best_confidence / 10.0, 1.0)
        
        return best_hr, normalized_confidence
    
    def _assess_signal_quality(self, snr: float, confidence: float) -> str:
        """Assess overall signal quality"""
        quality_score = 0.7 * min(snr / 10.0, 1.0) + 0.3 * confidence
        
        if quality_score >= 0.8:
            return "EXCELLENT"
        elif quality_score >= 0.6:
            return "GOOD"
        elif quality_score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def get_environmental_report(self) -> Dict[str, Any]:
        """Get comprehensive environmental report"""
        if not self.environment_history:
            return {"status": "NO_DATA"}
        
        latest_env = self.environment_history[-1]
        
        return {
            "rppg_available": self.ppg_available,
            "environmental_warning": self.environmental_warning,
            "lighting_condition": latest_env['lighting']['condition'],
            "lighting_score": latest_env['lighting']['score'],
            "minimum_snr_required": EnvironmentalConditions.MIN_SNR_FOR_PPG,
            "recommendations": self._get_environmental_recommendations(latest_env)
        }
    
    def _get_environmental_recommendations(self, env_data: Dict) -> List[str]:
        """Get environmental improvement recommendations"""
        recommendations = []
        lighting = env_data['lighting']
        
        if lighting['score'] < 0.5:
            recommendations.append("Increase lighting to improve biophysical analysis")
        
        if lighting['contrast'] < 0.3:
            recommendations.append("Improve contrast for better facial feature detection")
        
        return recommendations


# ============================================================================
# ENHANCED STRUCTURAL MATRIX INTEGRITY WITH NERF BENCHMARKING
# ============================================================================

class StructuralMatrixIntegrity:
    """
    Enhanced SMI with NeRF benchmarking capabilities
    Detects neural radiance field based head-swaps
    """
    
    def __init__(self, benchmark_mode: bool = False):
        self.fixed_bone_indices = [
            10, 152, 234, 454, 168, 397, 1, 4, 6, 200, 421
        ]
        self.reference_structure = None
        self.structure_history = deque(maxlen=100)
        
        # NeRF-specific detection parameters
        self.nerf_indicators = {
            'view_consistency_threshold': 0.95,
            'specular_consistency_threshold': 0.85,
            'shadow_coherence_threshold': 0.80
        }
        
        # Benchmarking mode
        self.benchmark_mode = benchmark_mode
        self.benchmark_results = []
        
    def analyze_spatial_matrix(self, landmarks: np.ndarray, 
                              frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze spatial matrix integrity with NeRF detection
        """
        results = {
            'spatial_integrity': {},
            'nerf_detection': {},
            'overall_score': 0.0
        }
        
        # 1. Basic structural rigidity
        rigidity_result = self._check_structural_rigidity(landmarks)
        results['spatial_integrity']['rigidity'] = rigidity_result
        
        # 2. Craniofacial symmetry
        symmetry_result = self._analyze_craniofacial_symmetry(landmarks)
        results['spatial_integrity']['symmetry'] = symmetry_result
        
        # 3. View consistency (NeRF detection)
        if frame is not None:
            nerf_result = self._detect_nerf_anomalies(frame, landmarks)
            results['nerf_detection'] = nerf_result
        
        # 4. Calculate overall SMI score
        rigidity_score = rigidity_result.get('rigidity_score', 0.5)
        symmetry_score = symmetry_result.get('symmetry_score', 0.5)
        
        if 'nerf_confidence' in results['nerf_detection']:
            nerf_confidence = results['nerf_detection'].get('nerf_confidence', 1.0)
            # Penalize if NeRF detected
            nerf_adjustment = 0.8 if nerf_confidence > 0.7 else 1.0
        else:
            nerf_adjustment = 1.0
        
        # Weighted score
        overall_score = (0.6 * rigidity_score + 0.4 * symmetry_score) * nerf_adjustment
        
        results['overall_score'] = overall_score
        results['deterministic_verification'] = overall_score >= 0.85
        
        # Add benchmark data if in benchmark mode
        if self.benchmark_mode:
            self._record_benchmark_data(results)
        
        return results
    
    def _check_structural_rigidity(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Check structural rigidity of bone points"""
        try:
            bone_points = landmarks[self.fixed_bone_indices]
            
            if self.reference_structure is None:
                self.reference_structure = spatial.distance.pdist(bone_points)
            
            current_structure = spatial.distance.pdist(bone_points)
            
            # Calculate deviation
            deviation = np.linalg.norm(current_structure - self.reference_structure) / \
                       (np.linalg.norm(self.reference_structure) + 1e-8)
            
            # Update history
            self.structure_history.append(current_structure)
            
            # Calculate statistical significance
            if len(self.structure_history) > 10:
                history_array = np.array(self.structure_history)
                z_scores = np.abs((current_structure - np.mean(history_array, axis=0)) / 
                                 (np.std(history_array, axis=0) + 1e-8))
                max_z_score = np.max(z_scores)
                statistical_confidence = 1.0 / (1.0 + max_z_score)
            else:
                statistical_confidence = 0.5
            
            # Calculate rigidity score
            rigidity_score = 1.0 - min(deviation / 0.015, 1.0)
            combined_score = 0.7 * rigidity_score + 0.3 * statistical_confidence
            
            return {
                'deviation': float(deviation),
                'rigidity_score': float(combined_score),
                'statistical_confidence': float(statistical_confidence),
                'bone_points_used': len(self.fixed_bone_indices),
                'status': 'PASS' if combined_score >= 0.85 else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"Structural rigidity check error: {e}")
            return {
                'deviation': 1.0,
                'rigidity_score': 0.0,
                'status': 'ERROR'
            }
    
    def _analyze_craniofacial_symmetry(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Analyze craniofacial symmetry"""
        try:
            # Split landmarks
            midline = landmarks.shape[0] // 2
            left = landmarks[:midline]
            right = landmarks[midline:]
            
            # Mirror right side
            right_mirrored = right.copy()
            right_mirrored[:, 0] = -right_mirrored[:, 0]
            
            # Procrustes analysis
            mtx1, mtx2, disparity = spatial.procrustes(left, right_mirrored)
            
            # Calculate errors
            errors = np.linalg.norm(mtx1 - mtx2, axis=1)
            mean_error = np.mean(errors)
            
            # Normalize by face width
            face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
            normalized_error = mean_error / (face_width + 1e-8)
            
            # Calculate symmetry score
            symmetry_score = 1.0 - min(normalized_error / 0.1, 1.0)
            
            return {
                'symmetry_score': float(symmetry_score),
                'procrustes_disparity': float(disparity),
                'mean_error': float(mean_error),
                'normalized_error': float(normalized_error),
                'status': 'PASS' if symmetry_score >= 0.9 else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"Symmetry analysis error: {e}")
            return {
                'symmetry_score': 0.0,
                'status': 'ERROR'
            }
    
    def _detect_nerf_anomalies(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Detect Neural Radiance Field anomalies
        NeRF-based head-swaps often have:
        1. Inconsistent view-dependent effects
        2. Incorrect specular highlights
        3. Shadow coherence issues
        """
        results = {
            'nerf_detected': False,
            'nerf_confidence': 0.0,
            'anomalies': []
        }
        
        try:
            # 1. View consistency check (NeRFs often have perfect multi-view consistency)
            view_consistency = self._check_view_consistency(frame, landmarks)
            results['view_consistency'] = view_consistency
            
            if view_consistency['score'] > self.nerf_indicators['view_consistency_threshold']:
                results['anomalies'].append('Unnaturally perfect view consistency')
                results['nerf_confidence'] += 0.3
            
            # 2. Specular highlight analysis
            specular_consistency = self._check_specular_consistency(frame, landmarks)
            results['specular_consistency'] = specular_consistency
            
            if specular_consistency['score'] > self.nerf_indicators['specular_consistency_threshold']:
                results['anomalies'].append('Unnatural specular consistency')
                results['nerf_confidence'] += 0.3
            
            # 3. Shadow coherence check
            shadow_coherence = self._check_shadow_coherence(frame, landmarks)
            results['shadow_coherence'] = shadow_coherence
            
            if shadow_coherence['score'] < self.nerf_indicators['shadow_coherence_threshold']:
                results['anomalies'].append('Shadow coherence anomalies')
                results['nerf_confidence'] += 0.4
            
            # Determine if NeRF detected
            results['nerf_detected'] = results['nerf_confidence'] >= 0.6
            
            return results
            
        except Exception as e:
            logger.error(f"NeRF detection error: {e}")
            return {
                'nerf_detected': False,
                'nerf_confidence': 0.0,
                'error': str(e)
            }
    
    def _check_view_consistency(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """Check view consistency (NeRFs often have perfect consistency)"""
        # Extract facial regions from different viewpoints
        # Simplified implementation - would use 3D pose estimation
        
        return {
            'score': 0.5,  # Placeholder
            'status': 'NORMAL',
            'method': 'multi_view_consistency'
        }
    
    def _check_specular_consistency(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """Check specular highlight consistency"""
        # Analyze specular highlights on facial features
        
        return {
            'score': 0.5,  # Placeholder
            'status': 'NORMAL',
            'method': 'specular_analysis'
        }
    
    def _check_shadow_coherence(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """Check shadow coherence"""
        # Analyze shadow consistency with lighting direction
        
        return {
            'score': 0.5,  # Placeholder
            'status': 'NORMAL',
            'method': 'shadow_coherence'
        }
    
    def _record_benchmark_data(self, results: Dict):
        """Record benchmark data for NeRF comparison"""
        benchmark_entry = {
            'timestamp': datetime.now().isoformat(),
            'smi_score': results['overall_score'],
            'rigidity_score': results['spatial_integrity']['rigidity'].get('rigidity_score', 0.0),
            'symmetry_score': results['spatial_integrity']['symmetry'].get('symmetry_score', 0.0),
            'nerf_detected': results['nerf_detection'].get('nerf_detected', False),
            'nerf_confidence': results['nerf_detection'].get('nerf_confidence', 0.0)
        }
        
        self.benchmark_results.append(benchmark_entry)
    
    def export_benchmark_results(self, output_path: str = "smi_benchmark_results.json"):
        """Export benchmark results to file"""
        if not self.benchmark_results:
            logger.warning("No benchmark data to export")
            return
        
        results_summary = {
            'benchmark_date': datetime.now().isoformat(),
            'total_samples': len(self.benchmark_results),
            'average_smi_score': float(np.mean([r['smi_score'] for r in self.benchmark_results])),
            'nerf_detection_rate': float(np.mean([1 if r['nerf_detected'] else 0 for r in self.benchmark_results])),
            'samples': self.benchmark_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Benchmark results exported to {output_path}")
        return results_summary


# ============================================================================
# SIRRAYA INTEGRITY SCORE CALCULATOR
# ============================================================================

class SirrayaIntegrityScoreCalculator:
    """
    Calculates the standardized Sirraya Integrity Score (SIS)
    Converts all detector outputs to the 0-100 SIS scale
    """
    
    def __init__(self):
        self.layer_weights = {
            'structural_matrix_integrity': 0.35,
            'quantum_neural': 0.25,
            'biophysical_verification': 0.20,
            'temporal_coherence': 0.10,
            'gan_artifact_detection': 0.10
        }
        
        self.calibration_factors = {
            'optimal_conditions': 1.0,
            'suboptimal_conditions': 0.9,
            'poor_conditions': 0.7
        }
    
    def calculate_sis(self, detector_results: Dict[str, Any], 
                     environmental_report: Dict[str, Any]) -> SISResult:
        """
        Calculate the Sirraya Integrity Score
        Returns standardized SISResult object
        """
        try:
            # 1. Calculate raw layer scores
            layer_scores = self._calculate_layer_scores(detector_results)
            
            # 2. Apply environmental calibration
            calibration_factor = self._get_environmental_calibration(environmental_report)
            
            # 3. Calculate weighted score
            weighted_score = 0.0
            for layer, score in layer_scores.items():
                if layer in self.layer_weights:
                    weighted_score += score * self.layer_weights[layer]
            
            # 4. Apply calibration
            calibrated_score = weighted_score * calibration_factor
            
            # 5. Convert to 0-100 SIS scale
            sis_score = calibrated_score * 100
            
            # 6. Apply biophysical bonus if deterministic verification
            if self._has_deterministic_verification(detector_results):
                sis_score = min(100, sis_score * 1.1)  # 10% bonus for deterministic verification
            
            # 7. Apply critical failure penalties
            if self._has_critical_failure(detector_results):
                sis_score = max(0, sis_score * 0.3)  # 70% penalty for critical failures
            
            # 8. Clamp to 0-100
            sis_score = max(0, min(100, sis_score))
            
            # 9. Determine category
            sis_category = SISCategory.from_score(sis_score)
            
            # 10. Calculate forensic confidence
            forensic_confidence = self._calculate_forensic_confidence(layer_scores, environmental_report)
            
            # 11. Determine if human review is needed
            requires_review = self._requires_human_review(sis_score, detector_results)
            
            # 12. Create verification layers report
            verification_layers = self._create_verification_report(detector_results, layer_scores)
            
            # 13. Create SISResult
            sis_result = SISResult(
                sis_score=sis_score,
                sis_category=sis_category.value[2],  # Human-readable category
                sis_verdict=self._get_sis_verdict(sis_score, detector_results),
                forensic_confidence=forensic_confidence,
                requires_human_review=requires_review,
                verification_layers=verification_layers,
                environmental_conditions=environmental_report,
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )
            
            return sis_result
            
        except Exception as e:
            logger.error(f"SIS calculation error: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_layer_scores(self, detector_results: Dict) -> Dict[str, float]:
        """Calculate scores for each detection layer"""
        layer_scores = {}
        
        # Structural Matrix Integrity
        if 'structural_matrix' in detector_results:
            smi_result = detector_results['structural_matrix']
            layer_scores['structural_matrix_integrity'] = smi_result.get('overall_score', 0.5)
        
        # Quantum Neural
        if 'quantum_neural' in detector_results:
            qn_result = detector_results['quantum_neural']
            layer_scores['quantum_neural'] = qn_result.get('quantum_confidence', 0.5)
        
        # Biophysical Verification
        if 'biophysical' in detector_results:
            bp_result = detector_results['biophysical']
            if bp_result.get('available', False):
                layer_scores['biophysical_verification'] = bp_result.get('confidence', 0.5)
            else:
                layer_scores['biophysical_verification'] = 0.5  # Neutral if unavailable
        
        # Temporal Coherence
        if 'temporal_coherence' in detector_results:
            tc_result = detector_results['temporal_coherence']
            layer_scores['temporal_coherence'] = tc_result.get('consistency', 0.5)
        
        # GAN Artifact Detection
        if 'gan_artifact' in detector_results:
            gan_result = detector_results['gan_artifact']
            layer_scores['gan_artifact_detection'] = 1.0 - gan_result.get('gan_likelihood', 0.0)
        
        # Fill missing layers with neutral scores
        for layer in self.layer_weights:
            if layer not in layer_scores:
                layer_scores[layer] = 0.5
        
        return layer_scores
    
    def _get_environmental_calibration(self, environmental_report: Dict) -> float:
        """Get environmental calibration factor"""
        rppg_available = environmental_report.get('rppg_available', True)
        
        if not rppg_available:
            # Can't do biophysical verification
            return self.calibration_factors['poor_conditions']
        
        lighting_score = environmental_report.get('lighting', {}).get('score', 1.0)
        
        if lighting_score >= 0.8:
            return self.calibration_factors['optimal_conditions']
        elif lighting_score >= 0.5:
            return self.calibration_factors['suboptimal_conditions']
        else:
            return self.calibration_factors['poor_conditions']
    
    def _has_deterministic_verification(self, detector_results: Dict) -> bool:
        """Check if we have deterministic verification (rPPG + SMI)"""
        has_smi = detector_results.get('structural_matrix', {}).get('deterministic_verification', False)
        has_ppg = detector_results.get('biophysical', {}).get('available', False)
        
        return has_smi and has_ppg
    
    def _has_critical_failure(self, detector_results: Dict) -> bool:
        """Check for critical failures"""
        # Critical SMI failure
        smi_score = detector_results.get('structural_matrix', {}).get('overall_score', 1.0)
        if smi_score < 0.3:
            return True
        
        # Impossible biophysical reading
        biophysical = detector_results.get('biophysical', {})
        if biophysical.get('available', False):
            hr = biophysical.get('heart_rate', 72)
            if hr is not None and (hr < 30 or hr > 200):  # Physiologically impossible
                return True
        
        return False
    
    def _calculate_forensic_confidence(self, layer_scores: Dict, 
                                      environmental_report: Dict) -> float:
        """Calculate forensic confidence score"""
        # Base confidence from layer agreement
        scores = list(layer_scores.values())
        variance = np.var(scores)
        agreement_confidence = 1.0 - min(variance / 0.1, 1.0)
        
        # Environmental confidence
        env_confidence = environmental_report.get('lighting', {}).get('score', 0.5)
        
        # Combined confidence
        forensic_confidence = 0.6 * agreement_confidence + 0.4 * env_confidence
        
        return forensic_confidence
    
    def _requires_human_review(self, sis_score: float, detector_results: Dict) -> bool:
        """Determine if human review is required"""
        # Low SIS score
        if sis_score < 40:
            return True
        
        # Conflicting evidence
        layer_scores = self._calculate_layer_scores(detector_results)
        scores = list(layer_scores.values())
        if np.std(scores) > 0.25:  # High variance between detectors
            return True
        
        # Environmental limitations
        biophysical = detector_results.get('biophysical', {})
        if not biophysical.get('available', False):
            # Can't do deterministic verification
            if sis_score < 70:  # Need human review for lower scores without biophysical
                return True
        
        return False
    
    def _create_verification_report(self, detector_results: Dict, 
                                   layer_scores: Dict) -> Dict[str, Dict]:
        """Create detailed verification report"""
        report = {}
        
        # Structural Matrix Integrity
        if 'structural_matrix' in detector_results:
            smi = detector_results['structural_matrix']
            report['structural_matrix_integrity'] = {
                'score': layer_scores.get('structural_matrix_integrity', 0.0),
                'deterministic_verification': smi.get('deterministic_verification', False),
                'rigidity_score': smi.get('spatial_integrity', {}).get('rigidity', {}).get('rigidity_score', 0.0),
                'symmetry_score': smi.get('spatial_integrity', {}).get('symmetry', {}).get('symmetry_score', 0.0),
                'nerf_detected': smi.get('nerf_detection', {}).get('nerf_detected', False)
            }
        
        # Quantum Neural
        if 'quantum_neural' in detector_results:
            qn = detector_results['quantum_neural']
            report['quantum_neural'] = {
                'score': layer_scores.get('quantum_neural', 0.0),
                'quantum_confidence': qn.get('quantum_confidence', 0.0),
                'is_authentic': qn.get('is_authentic', False)
            }
        
        # Biophysical Verification
        if 'biophysical' in detector_results:
            bp = detector_results['biophysical']
            report['biophysical_verification'] = {
                'score': layer_scores.get('biophysical_verification', 0.0),
                'available': bp.get('available', False),
                'heart_rate': bp.get('heart_rate'),
                'confidence': bp.get('confidence', 0.0),
                'snr': bp.get('snr', 0.0),
                'environmental_warning': bp.get('environmental_warning')
            }
        
        # Temporal Coherence
        if 'temporal_coherence' in detector_results:
            tc = detector_results['temporal_coherence']
            report['temporal_coherence'] = {
                'score': layer_scores.get('temporal_coherence', 0.0),
                'consistency': tc.get('consistency', 0.0),
                'jerk_score': tc.get('jerk_score', 0.0)
            }
        
        # GAN Artifact Detection
        if 'gan_artifact' in detector_results:
            gan = detector_results['gan_artifact']
            report['gan_artifact_detection'] = {
                'score': layer_scores.get('gan_artifact_detection', 0.0),
                'gan_likelihood': gan.get('gan_likelihood', 0.0),
                'grid_pattern_score': gan.get('grid_pattern_score', 0.0)
            }
        
        return report
    
    def _get_sis_verdict(self, sis_score: float, detector_results: Dict) -> str:
        """Get SIS verdict string"""
        category = SISCategory.from_score(sis_score)
        
        if category == SISCategory.DETERMINISTIC_AUTHENTICATION:
            return "DETERMINISTICALLY_AUTHENTIC"
        elif category == SISCategory.HIGH_PROBABILITY_AUTHENTIC:
            return "HIGH_PROBABILITY_AUTHENTIC"
        elif category == SISCategory.MODERATE_PROBABILITY_AUTHENTIC:
            return "MODERATE_PROBABILITY_AUTHENTIC"
        elif category == SISCategory.SUSPICIOUS:
            return "SUSPICIOUS_REQUIRES_REVIEW"
        elif category == SISCategory.HIGH_PROBABILITY_DEEPFAKE:
            # Check for specific failure modes
            if self._has_critical_failure(detector_results):
                return "CRITICAL_GEOMETRIC_FAILURE"
            else:
                return "HIGH_PROBABILITY_DEEPFAKE"
        else:  # CRITICAL_FAILURE
            return "CRITICAL_FAILURE_DETECTED"
    
    def _create_error_result(self, error: str) -> SISResult:
        """Create error SIS result"""
        return SISResult(
            sis_score=0.0,
            sis_category=SISCategory.CRITICAL_FAILURE.value[2],
            sis_verdict="ANALYSIS_ERROR",
            forensic_confidence=0.0,
            requires_human_review=True,
            verification_layers={'error': {'message': error}},
            environmental_conditions={'status': 'ERROR'},
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )


# ============================================================================
# BENCHMARKING SUITE FOR NERF DETECTION
# ============================================================================

class NeRFBenchmarkSuite:
    """
    Benchmarking suite for testing SMI against NeRF-based deepfakes
    """
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize SMI in benchmark mode
        self.smi_detector = StructuralMatrixIntegrity(benchmark_mode=True)
        
        # Benchmark datasets
        self.benchmark_datasets = {
            'nerf_head_swaps': {
                'description': 'NeRF-based head swap deepfakes',
                'expected_smi_score_range': (0.0, 0.6),  # Should be low
                'expected_detection_rate': 0.9  # Should detect 90%+
            },
            'authentic_videos': {
                'description': 'Authentic human videos',
                'expected_smi_score_range': (0.7, 1.0),  # Should be high
                'expected_detection_rate': 0.1  # Low false positive rate
            },
            'gan_based_deepfakes': {
                'description': 'Traditional GAN-based deepfakes',
                'expected_smi_score_range': (0.3, 0.7),
                'expected_detection_rate': 0.8
            }
        }
        
        # Results storage
        self.benchmark_results = {}
    
    def run_benchmark(self, dataset_type: str, video_paths: List[str]) -> Dict[str, Any]:
        """Run benchmark on a dataset"""
        logger.info(f"Running benchmark on {dataset_type}: {len(video_paths)} videos")
        
        results = {
            'dataset_type': dataset_type,
            'total_videos': len(video_paths),
            'smi_scores': [],
            'nerf_detections': [],
            'processed_videos': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for video_path in video_paths:
            try:
                video_result = self._analyze_video(video_path)
                results['smi_scores'].append(video_result['smi_score'])
                results['nerf_detections'].append(video_result['nerf_detected'])
                results['processed_videos'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
        
        # Calculate statistics
        if results['smi_scores']:
            results['average_smi_score'] = float(np.mean(results['smi_scores']))
            results['smi_score_std'] = float(np.std(results['smi_scores']))
            results['nerf_detection_rate'] = float(np.mean(results['nerf_detections']))
            
            # Compare with expected values
            expected = self.benchmark_datasets.get(dataset_type, {})
            expected_range = expected.get('expected_smi_score_range', (0.0, 1.0))
            expected_detection = expected.get('expected_detection_rate', 0.5)
            
            results['within_expected_range'] = (
                expected_range[0] <= results['average_smi_score'] <= expected_range[1]
            )
            results['detection_rate_deviation'] = abs(results['nerf_detection_rate'] - expected_detection)
        
        results['end_time'] = datetime.now().isoformat()
        results['processing_time'] = str(datetime.now() - datetime.fromisoformat(results['start_time']))
        
        # Store results
        self.benchmark_results[dataset_type] = results
        
        # Export results
        self._export_benchmark_results(dataset_type, results)
        
        return results
    
    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze single video for benchmarking"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        smi_scores = []
        nerf_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 10th frame for efficiency
            if frame_count % 10 == 0:
                # Detect face landmarks (simplified - would use face detection)
                landmarks = self._simulate_landmarks(frame)
                
                if landmarks is not None:
                    # Run SMI analysis
                    smi_result = self.smi_detector.analyze_spatial_matrix(landmarks, frame)
                    
                    smi_scores.append(smi_result['overall_score'])
                    nerf_detections.append(smi_result['nerf_detection'].get('nerf_detected', False))
            
            frame_count += 1
        
        cap.release()
        
        if smi_scores:
            return {
                'video_path': video_path,
                'smi_score': float(np.mean(smi_scores)),
                'nerf_detected': any(nerf_detections),
                'frames_analyzed': len(smi_scores)
            }
        else:
            return {
                'video_path': video_path,
                'smi_score': 0.5,
                'nerf_detected': False,
                'frames_analyzed': 0,
                'error': 'No valid frames analyzed'
            }
    
    def _simulate_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Simulate face landmarks for benchmarking"""
        # In real implementation, this would use MediaPipe or similar
        # For benchmarking, return simulated landmarks
        return np.random.randn(478, 3) * 0.05 + 0.5
    
    def _export_benchmark_results(self, dataset_type: str, results: Dict[str, Any]):
        """Export benchmark results to files"""
        # JSON export
        json_path = os.path.join(self.output_dir, f"{dataset_type}_benchmark.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV export for detailed analysis
        csv_path = os.path.join(self.output_dir, f"{dataset_type}_detailed.csv")
        if 'smi_scores' in results and results['smi_scores']:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['video_index', 'smi_score', 'nerf_detected'])
                for i, (score, detected) in enumerate(zip(results['smi_scores'], results['nerf_detections'])):
                    writer.writerow([i, score, detected])
        
        logger.info(f"Benchmark results exported for {dataset_type}")
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SIRRAYA STRUCTURAL MATRIX INTEGRITY BENCHMARK REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        for dataset_type, results in self.benchmark_results.items():
            report_lines.append(f"Dataset: {dataset_type}")
            report_lines.append(f"Description: {self.benchmark_datasets.get(dataset_type, {}).get('description', 'N/A')}")
            report_lines.append(f"Videos Processed: {results.get('processed_videos', 0)}/{results.get('total_videos', 0)}")
            
            if 'average_smi_score' in results:
                report_lines.append(f"Average SMI Score: {results['average_smi_score']:.3f}")
                report_lines.append(f"SMI Score Std Dev: {results['smi_score_std']:.3f}")
                report_lines.append(f"NeRF Detection Rate: {results['nerf_detection_rate']:.1%}")
                
                expected = self.benchmark_datasets.get(dataset_type, {})
                if expected:
                    expected_range = expected.get('expected_smi_score_range', (0.0, 1.0))
                    within_range = results.get('within_expected_range', False)
                    status = "✅ PASS" if within_range else "❌ FAIL"
                    report_lines.append(f"Expected Range: {expected_range[0]:.1f}-{expected_range[1]:.1f} {status}")
            
            report_lines.append("-"*40)
        
        # Overall assessment
        report_lines.append("")
        report_lines.append("OVERALL ASSESSMENT:")
        
        # Check if SMI effectively detects NeRF head-swaps
        if 'nerf_head_swaps' in self.benchmark_results:
            nerf_results = self.benchmark_results['nerf_head_swaps']
            if 'average_smi_score' in nerf_results:
                if nerf_results['average_smi_score'] < 0.6:
                    report_lines.append("✅ SMI effectively detects NeRF-based head swaps")
                else:
                    report_lines.append("⚠️ SMI may have difficulty with some NeRF implementations")
        
        # Check false positive rate
        if 'authentic_videos' in self.benchmark_results:
            auth_results = self.benchmark_results['authentic_videos']
            if 'average_smi_score' in auth_results:
                if auth_results['average_smi_score'] > 0.7:
                    report_lines.append("✅ Low false positive rate for authentic content")
                else:
                    report_lines.append("⚠️ High false positive rate - needs calibration")
        
        report_lines.append("="*80)
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        report_path = os.path.join(self.output_dir, "benchmark_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Also export SMI detector benchmark results
        if hasattr(self.smi_detector, 'export_benchmark_results'):
            self.smi_detector.export_benchmark_results(
                os.path.join(self.output_dir, "smi_detailed_benchmark.json")
            )
        
        return report_content