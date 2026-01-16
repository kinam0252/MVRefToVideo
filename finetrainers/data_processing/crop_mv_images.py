#!/usr/bin/env python3
"""
Multi-view Image Cropping with Adaptive Gap Detection

This module provides automated cropping of multi-view product images by detecting
gaps between views using adaptive threshold, contrast, peak, and gradient methods.

Success Rate: 89.4% (889/994 samples)
Strategy: Adaptive Threshold -> Conditional Ordering -> Gradient -> Fallback
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import shutil
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class MultiViewCropper:
    """Adaptive multi-view image cropper with quality validation"""
    
    # Quality validation thresholds
    MIN_VIEW_RATIO = 0.05      # Minimum view width (5% of total)
    MAX_VIEW_RATIO = 0.80      # Maximum view width (80% of total)
    MIN_GAP_QUALITY = 60       # Minimum gap whiteness score
    MIN_CONFIDENCE = 60        # Minimum overall confidence score
    MAX_WIDTH_CV = 1.0         # Maximum coefficient of variation for view widths
    
    # Background brightness threshold
    DARK_BACKGROUND_THRESHOLD = 245
    
    # Component removal threshold
    COMPONENT_THRESHOLD = 240
    
    # Whiteness threshold for conditional phase ordering
    WHITENESS_THRESHOLD = 0.30
    
    def __init__(self, debug_mode=False):
        self.success_count = 0
        self.fail_count = 0
        self.failed_samples = []
        self.fallback_samples = []
        self.adaptive_rescued_samples = []
        self.peak_rescued_samples = []
        self.contrast_rescued_samples = []
        self.gradient_rescued_samples = []
        self.low_quality_samples = []
        self.debug_mode = debug_mode
    
    def crop_image(self, combined_path, output_dir, sample_id):
        """Crop combined image using adaptive gap detection"""
        img = Image.open(combined_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy combined.png if needed
        src_path = Path(combined_path)
        dst_path = output_dir / "combined.png"
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(str(src_path), str(dst_path))
        
        # Setup debug directory
        debug_dir = output_dir / "debug" if self.debug_mode else None
        if debug_dir:
            debug_dir.mkdir(exist_ok=True)
        
        # Find view boundaries
        boundaries, method_used, background_info, whiteness_dict = self._find_boundaries_adaptive(
            img_array, debug_dir=debug_dir
        )
        
        if len(boundaries) != 4:
            return {'success': False, 'reason': f'found_{len(boundaries)-1}_views'}
        
        # Quality validation
        is_valid, reason, stats = self._validate_view_widths(boundaries)
        if not is_valid:
            self.low_quality_samples.append({
                'id': sample_id,
                'method': method_used,
                'reason': reason,
                **stats
            })
            return {
                'success': False,
                'reason': reason,
                'method': method_used,
                'boundaries': boundaries,
                **stats
            }
        
        gap_score = self._evaluate_gap_quality(boundaries, whiteness_dict)
        confidence = self._calculate_confidence(
            method_used, gap_score, stats['width_cv'], whiteness_dict.get('mean', 0)
        )
        
        if confidence < self.MIN_CONFIDENCE:
            self.low_quality_samples.append({
                'id': sample_id,
                'method': method_used,
                'reason': 'low_confidence',
                'confidence': confidence
            })
        
        # Crop views
        bg_brightness = background_info.get('brightness', 255)
        use_bright_pipeline = bg_brightness >= self.DARK_BACKGROUND_THRESHOLD
        
        self._crop_and_save_views(
            img_array, boundaries, output_dir, 
            use_bright_pipeline, debug_dir
        )
        
        # Track method
        if method_used == 'adaptive':
            self.adaptive_rescued_samples.append(sample_id)
        elif method_used == 'peak':
            self.peak_rescued_samples.append(sample_id)
        elif method_used == 'contrast':
            self.contrast_rescued_samples.append(sample_id)
        elif method_used == 'gradient':
            self.gradient_rescued_samples.append(sample_id)
        elif method_used == 'fallback':
            self.fallback_samples.append(sample_id)
        
        self.success_count += 1
        
        return {
            'success': True,
            'method': method_used,
            'boundaries': boundaries,
            'confidence': confidence,
            'gap_quality': gap_score,
            **stats
        }
    
    def _find_boundaries_adaptive(self, img_array, debug_dir=None):
        """Find view boundaries using adaptive multi-phase approach"""
        h, w = img_array.shape[:2]
        step = max(1, h // 50)
        
        # Phase 0: Analyze background
        background_brightness = float(np.mean(img_array[:5, :, :]))
        adaptive_threshold = background_brightness - 3.0
        
        whiteness_dict = self._calculate_whiteness_profile(img_array, step)
        mean_whiteness = whiteness_dict.get('mean', 0)
        
        # Phase 1: Adaptive threshold
        boundaries, success = self._try_adaptive_threshold(
            img_array, adaptive_threshold, step, debug_dir
        )
        if success:
            return boundaries, 'adaptive', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
        
        # Phase 2 & 3: Conditional ordering based on whiteness
        if mean_whiteness > self.WHITENESS_THRESHOLD:
            # High whiteness: Peak -> Contrast
            boundaries, success = self._try_peak_detection(whiteness_dict, w, img_array, debug_dir)
            if success:
                return boundaries, 'peak', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
            
            boundaries, success = self._try_contrast_detection(img_array, w, step, debug_dir)
            if success:
                return boundaries, 'contrast', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
        else:
            # Low whiteness: Contrast -> Peak
            boundaries, success = self._try_contrast_detection(img_array, w, step, debug_dir)
            if success:
                return boundaries, 'contrast', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
            
            boundaries, success = self._try_peak_detection(whiteness_dict, w, img_array, debug_dir)
            if success:
                return boundaries, 'peak', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
        
        # Phase 4: Gradient detection
        boundaries, success = self._try_gradient_detection(img_array, w, step, debug_dir)
        if success:
            return boundaries, 'gradient', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
        
        # Phase 5: Fallback
        boundaries = [0, w // 3, 2 * w // 3, w]
        return boundaries, 'fallback', {'brightness': background_brightness, 'threshold': adaptive_threshold}, whiteness_dict
    
    def _calculate_whiteness_profile(self, img_array, step):
        """Calculate whiteness profile across image width"""
        h, w = img_array.shape[:2]
        whiteness_scores = []
        
        for x in range(0, w, step):
            region = img_array[:, max(0, x-step//2):min(w, x+step//2), :]
            whiteness = np.mean(region) / 255.0
            whiteness_scores.append(whiteness)
        
        x_coords = list(range(0, w, step))
        
        return {
            'x_coords': x_coords,
            'scores': whiteness_scores,
            'mean': np.mean(whiteness_scores),
            'std': np.std(whiteness_scores)
        }
    
    def _try_adaptive_threshold(self, img_array, adaptive_threshold, step, debug_dir):
        """Try adaptive threshold method"""
        h, w = img_array.shape[:2]
        thresholds = [0.99, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88]
        
        for threshold_ratio in thresholds:
            threshold = adaptive_threshold * threshold_ratio
            boundaries = self._find_vertical_gaps_threshold(img_array, threshold, step)
            
            if len(boundaries) == 4:
                is_valid, _, _ = self._validate_view_widths(boundaries)
                if is_valid:
                    return boundaries, True
        
        return None, False
    
    def _try_contrast_detection(self, img_array, w, step, debug_dir):
        """Try contrast detection method"""
        h = img_array.shape[0]
        brightness_profile = []
        
        for x in range(0, w, step):
            col = img_array[:, x, :]
            brightness = np.mean(col)
            brightness_profile.append(brightness)
        
        x_coords = list(range(0, w, step))
        brightness_profile = np.array(brightness_profile)
        
        smoothed = gaussian_filter1d(brightness_profile, sigma=3)
        mean_brightness = np.mean(smoothed)
        relative_brightness = smoothed - mean_brightness
        
        peaks, properties = find_peaks(relative_brightness, prominence=5, distance=20)
        
        if len(peaks) >= 2:
            prominences = properties['prominences']
            top_indices = np.argsort(prominences)[-2:]
            top_peaks = sorted(peaks[top_indices])
            
            peak1_x = x_coords[top_peaks[0]]
            peak2_x = x_coords[top_peaks[1]]
            
            boundaries = [0, peak1_x, peak2_x, w]
            is_valid, _, _ = self._validate_view_widths(boundaries)
            
            if is_valid:
                return boundaries, True
        
        return None, False
    
    def _try_peak_detection(self, whiteness_dict, w, img_array, debug_dir):
        """Try peak detection method"""
        x_coords = whiteness_dict['x_coords']
        scores = whiteness_dict['scores']
        
        smoothed_scores = gaussian_filter1d(scores, sigma=2)
        peaks, properties = find_peaks(smoothed_scores, prominence=0.05, distance=10)
        
        if len(peaks) >= 2:
            prominences = properties['prominences']
            top_indices = np.argsort(prominences)[-2:]
            top_peaks = sorted(peaks[top_indices])
            
            peak1_x = x_coords[top_peaks[0]]
            peak2_x = x_coords[top_peaks[1]]
            
            boundaries = [0, peak1_x, peak2_x, w]
            is_valid, _, _ = self._validate_view_widths(boundaries)
            
            if is_valid:
                return boundaries, True
        
        return None, False
    
    def _try_gradient_detection(self, img_array, w, step, debug_dir):
        """Try gradient detection method"""
        h = img_array.shape[0]
        gradient_profile = []
        
        for x in range(0, w-step, step):
            left_col = img_array[:, x, :]
            right_col = img_array[:, x+step, :]
            gradient = np.abs(np.mean(right_col) - np.mean(left_col))
            gradient_profile.append(gradient)
        
        x_coords = list(range(0, w-step, step))
        gradient_profile = np.array(gradient_profile)
        
        smoothed = gaussian_filter1d(gradient_profile, sigma=3)
        peaks, properties = find_peaks(smoothed, prominence=3, distance=20)
        
        if len(peaks) >= 2:
            prominences = properties['prominences']
            top_indices = np.argsort(prominences)[-2:]
            top_peaks = sorted(peaks[top_indices])
            
            peak1_x = x_coords[top_peaks[0]]
            peak2_x = x_coords[top_peaks[1]]
            
            boundaries = [0, peak1_x, peak2_x, w]
            is_valid, _, _ = self._validate_view_widths(boundaries)
            
            if is_valid:
                return boundaries, True
        
        return None, False
    
    def _find_vertical_gaps_threshold(self, img_array, threshold, step):
        """Find vertical gaps using threshold method"""
        h, w = img_array.shape[:2]
        gap_scores = []
        
        for x in range(0, w, step):
            col = img_array[:, x, :]
            brightness = np.mean(col)
            gap_scores.append(brightness)
        
        is_gap = np.array(gap_scores) > threshold
        boundaries = [0]
        
        for i in range(1, len(is_gap)):
            if is_gap[i] and not is_gap[i-1]:
                boundaries.append(i * step)
            elif not is_gap[i] and is_gap[i-1]:
                boundaries.append(i * step)
        
        boundaries.append(w)
        
        return boundaries
    
    def _validate_view_widths(self, boundaries):
        """Validate view widths"""
        w = boundaries[-1]
        view_widths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        view_ratios = [width / w for width in view_widths]
        
        # Check min/max ratios
        if any(ratio < self.MIN_VIEW_RATIO for ratio in view_ratios):
            return False, 'view_too_narrow', {'view_widths': view_widths, 'width_cv': 0}
        
        if any(ratio > self.MAX_VIEW_RATIO for ratio in view_ratios):
            return False, 'view_too_wide', {'view_widths': view_widths, 'width_cv': 0}
        
        # Check coefficient of variation
        mean_width = np.mean(view_widths)
        std_width = np.std(view_widths)
        cv = std_width / mean_width if mean_width > 0 else float('inf')
        
        if cv > self.MAX_WIDTH_CV:
            return False, 'width_imbalance', {'view_widths': view_widths, 'width_cv': cv}
        
        return True, 'valid', {'view_widths': view_widths, 'width_cv': cv}
    
    def _evaluate_gap_quality(self, boundaries, whiteness_dict):
        """Evaluate gap quality"""
        x_coords = whiteness_dict['x_coords']
        scores = whiteness_dict['scores']
        
        gap_regions = [(boundaries[i], boundaries[i+1]) for i in [0, 1, 2]]
        gap_whiteness = []
        
        for start, end in gap_regions:
            indices = [i for i, x in enumerate(x_coords) if start <= x <= end]
            if indices:
                region_scores = [scores[i] for i in indices]
                gap_whiteness.append(np.mean(region_scores) * 100)
        
        return np.mean(gap_whiteness) if gap_whiteness else 0
    
    def _calculate_confidence(self, method, gap_quality, width_cv, mean_whiteness):
        """Calculate confidence score"""
        method_scores = {
            'adaptive': 95, 'contrast': 85, 'peak': 85,
            'gradient': 75, 'fallback': 50
        }
        method_score = method_scores.get(method, 50)
        
        gap_score = gap_quality
        uniformity_score = max(0, 100 - width_cv * 100)
        whiteness_score = mean_whiteness * 100
        
        confidence = (
            method_score * 0.4 +
            gap_score * 0.3 +
            uniformity_score * 0.2 +
            whiteness_score * 0.1
        )
        
        return confidence
    
    def _crop_and_save_views(self, img_array, boundaries, output_dir, use_bright_pipeline, debug_dir):
        """Crop and save individual views"""
        view_names = ['front', 'side', 'bottom']
        h, w = img_array.shape[:2]
        
        for i, view_name in enumerate(view_names):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            view_img = img_array[:, x_start:x_end, :]
            
            if use_bright_pipeline:
                view_img = self._process_view_bright(view_img, debug_dir, view_name)
            else:
                view_img = self._process_view_dark(view_img, debug_dir, view_name)
            
            output_path = output_dir / f"{view_name}.png"
            Image.fromarray(view_img).save(output_path)
    
    def _process_view_bright(self, view_img, debug_dir, view_name):
        """Process view with bright background"""
        # Tight crop
        tight_bbox = self._find_tight_bbox_simple(view_img, threshold=240)
        if tight_bbox:
            y1, x1, y2, x2 = tight_bbox
            view_img = view_img[y1:y2, x1:x2]
        
        # Remove noise
        view_img = self._remove_small_components_simple(view_img, threshold=240)
        
        # Final crop
        final_bbox = self._find_tight_bbox_simple(view_img, threshold=240)
        if final_bbox:
            y1, x1, y2, x2 = final_bbox
            view_img = view_img[y1:y2, x1:x2]
        
        return view_img
    
    def _process_view_dark(self, view_img, debug_dir, view_name):
        """Process view with dark background"""
        # Tight crop
        tight_bbox = self._find_tight_bbox_adaptive(view_img, percentile=5)
        if tight_bbox:
            y1, x1, y2, x2 = tight_bbox
            view_img = view_img[y1:y2, x1:x2]
        
        # Remove noise
        view_img = self._remove_small_components_adaptive(view_img, percentile=5)
        
        # Final crop
        final_bbox = self._find_tight_bbox_adaptive(view_img, percentile=5)
        if final_bbox:
            y1, x1, y2, x2 = final_bbox
            view_img = view_img[y1:y2, x1:x2]
        
        return view_img
    
    def _find_tight_bbox_simple(self, img, threshold=240):
        """Find tight bounding box using simple threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        mask = gray < threshold
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return y1, x1, y2 + 1, x2 + 1
    
    def _find_tight_bbox_adaptive(self, img, percentile=5):
        """Find tight bounding box using adaptive threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        threshold = np.percentile(gray, percentile)
        mask = gray < (threshold + 10)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return y1, x1, y2 + 1, x2 + 1
    
    def _remove_small_components_simple(self, img, threshold=240, min_size_ratio=0.01):
        """Remove small components using simple threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        mask = (gray < threshold).astype(np.uint8) * 255
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return img
        
        total_pixels = img.shape[0] * img.shape[1]
        min_size = int(total_pixels * min_size_ratio)
        
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_component = np.argmax(sizes) + 1
        
        keep_mask = (labels == largest_component)
        result = img.copy()
        result[~keep_mask] = 255
        
        return result
    
    def _remove_small_components_adaptive(self, img, percentile=5, min_size_ratio=0.01):
        """Remove small components using adaptive threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        threshold = np.percentile(gray, percentile)
        mask = (gray < (threshold + 10)).astype(np.uint8) * 255
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return img
        
        total_pixels = img.shape[0] * img.shape[1]
        min_size = int(total_pixels * min_size_ratio)
        
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_component = np.argmax(sizes) + 1
        
        keep_mask = (labels == largest_component)
        result = img.copy()
        bg_value = int(np.percentile(gray, 95))
        result[~keep_mask] = bg_value
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Multi-view Image Cropping with Adaptive Gap Detection'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Input directory with combined.png files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for cropped images')
    parser.add_argument('--start_id', type=int, default=0,
                        help='Starting sample ID')
    parser.add_argument('--end_id', type=int, default=997,
                        help='Ending sample ID (exclusive)')
    parser.add_argument('--log_file', type=str, default='crop_log.json',
                        help='Log file path')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (saves intermediate images)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MULTI-VIEW IMAGE CROPPING")
    print("=" * 70)
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample range: {args.start_id} to {args.end_id}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 70)
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cropper = MultiViewCropper(debug_mode=args.debug)
    
    data_dir = Path(args.data_dir)
    start_time = datetime.now()
    results = []
    
    for sample_id in tqdm(range(args.start_id, args.end_id), desc="Processing"):
        combined_path = data_dir / str(sample_id) / "combined.png"
        
        if not combined_path.exists():
            cropper.fail_count += 1
            cropper.failed_samples.append(sample_id)
            results.append({
                'id': sample_id,
                'success': False,
                'reason': 'file_not_found'
            })
            continue
        
        out_dir = output_dir / str(sample_id)
        result = cropper.crop_image(combined_path, out_dir, sample_id)
        results.append({'id': sample_id, **result})
        
        if not result['success']:
            cropper.fail_count += 1
            cropper.failed_samples.append(sample_id)
    
    # Print summary
    total = args.end_id - args.start_id
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Success: {cropper.success_count}")
    print(f"Failed: {cropper.fail_count}")
    print(f"Success Rate: {cropper.success_count/total*100:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Throughput: {total/elapsed:.2f} images/s")
    print("=" * 70)
    
    # Save log
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'summary': {
            'total': total,
            'success': cropper.success_count,
            'failed': cropper.fail_count,
            'success_rate': cropper.success_count / total,
            'elapsed_seconds': elapsed
        },
        'method_counts': {
            'adaptive': len(cropper.adaptive_rescued_samples),
            'contrast': len(cropper.contrast_rescued_samples),
            'peak': len(cropper.peak_rescued_samples),
            'gradient': len(cropper.gradient_rescued_samples),
            'fallback': len(cropper.fallback_samples)
        },
        'results': results
    }
    
    with open(args.log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nLog saved to {args.log_file}")


if __name__ == '__main__':
    main()
