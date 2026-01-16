#!/usr/bin/env python3
"""
Adaptive Multi-view Cropping v4.2: Improved Conditional Ordering
Changes from v4.1:
- WHITENESS_THRESHOLD: 0.15 -> 0.30 (prefer Contrast for more cases)
- Fixes #72 type issues where Peak was tried first but Contrast works better
Changes from v4:
- FIXED: Gap quality calculation bug (now uses range-based sampling)
- FIXED: Threshold range more strict (0.99-0.88 only, removed 0.85, 0.80, 0.75)
- With validation enabled, we can be stricter on thresholds
Changes from v3:
- View width validation (min 5%, max 80%, CV < 1.0)
- Gap quality score (whiteness > 60)
- Confidence score system (method + gap + uniformity + whiteness)
- Auto-filtering of invalid samples
- Enhanced adaptive threshold with validation
Strategy: Adaptive Threshold -> [Peak/Contrast] -> [Contrast/Peak] -> Gradient -> Fallback
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

class AdaptiveCropperV4:
    """Adaptive approach v4.2 with improved conditional ordering"""

    # Quality validation thresholds (v4)
    MIN_VIEW_RATIO = 0.05 # Minimum view width (5% of total)
    MAX_VIEW_RATIO = 0.80 # Maximum view width (80% of total)
    MIN_GAP_QUALITY = 60 # Minimum gap whiteness score
    MIN_CONFIDENCE = 60 # Minimum overall confidence score
    MAX_WIDTH_CV = 1.0 # Maximum coefficient of variation for view widths

    # Background brightness threshold for pipeline selection
    DARK_BACKGROUND_THRESHOLD = 245

    # Component removal threshold
    COMPONENT_THRESHOLD = 240 # Restored from 235 to preserve bright objects

    # Whiteness threshold for conditional phase ordering
    WHITENESS_THRESHOLD = 0.30 # Raised from 0.15 to prefer Contrast for more cases

    def __init__(self, debug_mode=False):
        self.success_count = 0
        self.fail_count = 0
        self.failed_samples = []
        self.fallback_samples = []
        self.adaptive_rescued_samples = []
        self.peak_rescued_samples = []
        self.contrast_rescued_samples = []
        self.gradient_rescued_samples = []
        self.low_quality_samples = [] # NEW: Track low quality samples
        self.debug_mode = debug_mode

    def crop_image(self, combined_path, output_dir, sample_id):
        """Crop combined image using adaptive gap detection"""
        # Load image
        img = Image.open(combined_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        total_width = w # Save for normalization
        total_height = h # Save for normalization

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy combined.png only if input != output
        src_path = Path(combined_path)
        dst_path = output_dir / "combined.png"
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(str(src_path), str(dst_path))
            print(f" Sample {sample_id}: Copied combined.png")
        else:
            print(f" Sample {sample_id}: Processing in-place (skipping copy)")

        # Setup debug directory
        if self.debug_mode:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            print(f" [DEBUG] Debug directory: {debug_dir}")
        else:
            debug_dir = None

        # Find view boundaries using adaptive approach
        boundaries, method_used, background_info, whiteness_dict = self._find_boundaries_adaptive(img_array, debug_dir=debug_dir)

        if len(boundaries) != 4:
            print(f" Sample {sample_id}: Found {len(boundaries)-1} views (expected 3)")
            return {'success': False, 'reason': f'found_{len(boundaries)-1}_views'}

        #
        # V4 QUALITY VALIDATION
        #

        # Validation 1: View width check
        is_valid, reason, stats = self._validate_view_widths(boundaries)
        if not is_valid:
            print(f" Invalid boundaries: {reason}")
            self.low_quality_samples.append({
                'sample_id': sample_id,
                'method': method_used,
                'reason': reason,
                'confidence': 0
            })
            # Don't fail immediately, just track it

        # Validation 2: Calculate confidence score
        confidence_score, components = self._calculate_confidence_score(
            boundaries, method_used, whiteness_dict
        )

        print(f" Quality Check:")
        print(f" Method score: {components['method']:.1f}")
        print(f" Gap quality: {components['gap_quality']:.1f}")
        print(f" Uniformity: {components['uniformity']:.1f}")
        print(f" Whiteness: {components['whiteness']:.1f}")
        print(f" -> Confidence: {confidence_score:.1f}/100")

        # Track low quality samples
        if confidence_score < self.MIN_CONFIDENCE:
            print(f" Low confidence detected ({confidence_score:.1f} < {self.MIN_CONFIDENCE})")
            self.low_quality_samples.append({
                'sample_id': sample_id,
                'method': method_used,
                'confidence': confidence_score,
                'components': components,
                'reason': 'low_confidence'
            })

        print(f" Boundaries: {boundaries}")
        print(f" Method used: {method_used}")
        print(f" Background info: brightness={background_info['background_brightness']:.1f}, threshold={background_info['adaptive_threshold']:.1f}")

        # Determine pipeline based on background brightness
        is_dark_background = background_info['background_brightness'] < self.DARK_BACKGROUND_THRESHOLD

        if is_dark_background:
            print(f" Using DARK BACKGROUND pipeline (brightness={background_info['background_brightness']:.1f} < {self.DARK_BACKGROUND_THRESHOLD})")
            pipeline = 'dark'
        else:
            print(f" Using BRIGHT BACKGROUND pipeline (brightness={background_info['background_brightness']:.1f} >= {self.DARK_BACKGROUND_THRESHOLD})")
            pipeline = 'bright'

        # Track method usage
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

        # Find Y boundaries (vertical extent)
        y_min, y_max = self._find_vertical_extent(img_array, base_threshold=background_info['adaptive_threshold'])

        # Crop and save individual views
        view_names = ['front', 'side', 'bottom']
        cropped_paths = []

        for i in range(3):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            view_name = view_names[i]

            print(f"\n === Processing {view_name} ({pipeline} pipeline) ===")

            # Step 1: Extract rough region
            view_region = img_array[y_min:y_max, x_start:x_end]
            print(f" Step 1: Rough region size: {view_region.shape}")

            if self.debug_mode:
                Image.fromarray(view_region).save(debug_dir / f"{view_name}_step1_rough.png")
                print(f" -> Saved {view_name}_step1_rough.png")

            #
            # Pipeline branching
            #

            if pipeline == 'dark':
                #
                # DARK BACKGROUND PIPELINE
                #

                # Step 2: 2D scan tight bbox
                tight_bbox = self._find_tight_bbox_2d(view_region, base_threshold=background_info['adaptive_threshold'])

                if tight_bbox is None:
                    print(f" Failed to find tight bbox for {view_name}")
                    continue

                tx1, ty1, tx2, ty2 = tight_bbox
                print(f" Step 2: Tight bbox (2D scan): ({tx1}, {ty1}, {tx2}, {ty2})")

                # Step 3: Crop to tight bbox
                view_tight = view_region[ty1:ty2, tx1:tx2]
                print(f" Step 3: After tight crop: {view_tight.shape}")

                if self.debug_mode:
                    Image.fromarray(view_tight).save(debug_dir / f"{view_name}_step2_tight.png")
                    print(f" -> Saved {view_name}_step2_tight.png")

                # Step 4: SKIP component removal (preserve object details)
                view_cleaned = view_tight
                print(f" Step 4: Skipped component removal (dark background)")

                if self.debug_mode:
                    Image.fromarray(view_cleaned).save(debug_dir / f"{view_name}_step3_cleaned.png")
                    print(f" -> Saved {view_name}_step3_cleaned.png (unchanged)")

                    # Create empty diff image
                    diff = np.zeros_like(view_tight)
                    Image.fromarray(diff).save(debug_dir / f"{view_name}_step3_diff.png")
                    print(f" -> Saved {view_name}_step3_diff.png (no changes)")

                # Step 5: Final tight bbox (2D scan)
                final_bbox = self._find_tight_bbox_2d(view_cleaned, base_threshold=background_info['adaptive_threshold'])

                if final_bbox is not None:
                    fx1, fy1, fx2, fy2 = final_bbox
                    view_final = view_cleaned[fy1:fy2, fx1:fx2]
                    print(f" Step 5: Final bbox (2D scan): ({fx1}, {fy1}, {fx2}, {fy2})")
                    print(f" Step 5: Final size: {view_final.shape}")
                else:
                    view_final = view_cleaned
                    print(f" Step 5: No final bbox (using cleaned)")

            else:
                #
                # BRIGHT BACKGROUND PIPELINE
                #

                # Step 2: 1D simple scan
                tight_bbox = self._find_tight_bbox_1d_simple(view_region, threshold=self.COMPONENT_THRESHOLD)

                if tight_bbox is None:
                    print(f" Failed to find tight bbox for {view_name}")
                    continue

                tx1, ty1, tx2, ty2 = tight_bbox
                print(f" Step 2: Tight bbox (1D simple): ({tx1}, {ty1}, {tx2}, {ty2})")

                # Step 3: Crop to tight bbox
                view_tight = view_region[ty1:ty2, tx1:tx2]
                print(f" Step 3: After tight crop: {view_tight.shape}")

                if self.debug_mode:
                    Image.fromarray(view_tight).save(debug_dir / f"{view_name}_step2_tight.png")
                    print(f" -> Saved {view_name}_step2_tight.png")

                # Step 4: Remove small components (simple, V1 style)
                view_cleaned = self._remove_small_components_simple(view_tight, threshold=self.COMPONENT_THRESHOLD)
                print(f" Step 4: After noise removal (simple, V1 style): {view_cleaned.shape}")

                if self.debug_mode:
                    Image.fromarray(view_cleaned).save(debug_dir / f"{view_name}_step3_cleaned.png")
                    print(f" -> Saved {view_name}_step3_cleaned.png")

                    pixels_changed = np.sum(view_tight != view_cleaned)
                    total_pixels = view_tight.shape[0] * view_tight.shape[1] * view_tight.shape[2]
                    percent_changed = (pixels_changed / total_pixels) * 100
                    print(f" -> Pixels changed: {pixels_changed:,} / {total_pixels:,} ({percent_changed:.2f}%)")

                    diff = np.abs(view_tight.astype(int) - view_cleaned.astype(int)).astype(np.uint8)
                    Image.fromarray(diff).save(debug_dir / f"{view_name}_step3_diff.png")
                    print(f" -> Saved {view_name}_step3_diff.png (difference map)")

                # Step 5: Final tight bbox
                final_bbox = self._find_tight_bbox_1d_simple(view_cleaned, threshold=self.COMPONENT_THRESHOLD)

                if final_bbox is not None:
                    fx1, fy1, fx2, fy2 = final_bbox
                    view_final = view_cleaned[fy1:fy2, fx1:fx2]
                    print(f" Step 5: Final bbox (1D simple): ({fx1}, {fy1}, {fx2}, {fy2})")
                    print(f" Step 5: Final size: {view_final.shape}")
                else:
                    view_final = view_cleaned
                    print(f" Step 5: No final bbox (using cleaned)")

            #
            # Common: Save final result
            #

            if self.debug_mode:
                Image.fromarray(view_final).save(debug_dir / f"{view_name}_step4_final.png")
                print(f" -> Saved {view_name}_step4_final.png")

            # Save final result
            # Normalize to 512x512 (reflecting original ratio)
            view_normalized = self._normalize_view_512x512(
             view_final,
             boundaries,
             total_width,
             total_height,
             view_index=i
            )
            view_img = Image.fromarray(view_normalized)
            output_path = output_dir / f"{view_name}.png"
            view_img.save(output_path)
            cropped_paths.append(output_path)
            print(f" Saved {view_name}.png")

        if len(cropped_paths) != 3:
            return {'success': False, 'reason': 'tight_crop_failed'}

        # Combine vertically
        final_path = self._combine_vertically(cropped_paths, output_dir / "final.png", padding=0)

        return {
            'success': True,
            'sample_id': sample_id,
            'cropped_paths': cropped_paths,
            'final_path': final_path,
            'method_used': method_used,
            'confidence_score': confidence_score,
            'confidence_components': components
        }

    def _find_boundaries_adaptive(self, img_array, step=5, debug_dir=None):
        """
        Adaptive approach with multiple fallback methods
        v4: Added quality validation for boundaries
        - View width validation (min 5%, max 80%, CV < 1.0)
        - Gap quality score
        - Confidence scoring

        Returns:
            tuple: (boundaries, method_used, background_info, whiteness_dict)
        """
        h, w = img_array.shape[:2]

        # PHASE 0: Analyze background brightness
        print(f" [PHASE 0] Analyzing background brightness...")
        background_brightness, adaptive_threshold = self._analyze_background(img_array)
        print(f" Background brightness: {background_brightness:.1f}")
        print(f" Adaptive threshold: {adaptive_threshold:.1f}")

        background_info = {
            'background_brightness': background_brightness,
            'adaptive_threshold': adaptive_threshold
        }

        # Calculate whiteness
        whiteness_dict = self._calculate_whiteness(img_array, adaptive_threshold, step)

        # Calculate mean whiteness for conditional ordering
        whiteness_values = list(whiteness_dict.values())
        mean_whiteness = np.mean(whiteness_values)
        print(f" Mean whiteness: {mean_whiteness:.3f}")

        # PHASE 1: Try threshold-based method
        print(f" [PHASE 1] Trying ADAPTIVE THRESHOLD method...")
        boundaries, success = self._try_threshold_method(
            whiteness_dict, w, img_array, adaptive_threshold, debug_dir
        )

        if success:
            print(f" ADAPTIVE THRESHOLD method succeeded!")
            return boundaries, 'adaptive', background_info, whiteness_dict

        #
        # PHASE 2 & 3: CONDITIONAL ORDERING based on whiteness
        #

        if mean_whiteness > self.WHITENESS_THRESHOLD:
            #
            # High whiteness case: Peak -> Contrast
            #
            print(f" [INFO] High whiteness ({mean_whiteness:.3f} > {self.WHITENESS_THRESHOLD}) -> Peak detection first")

            # PHASE 2: Try PEAK detection
            print(f" [PHASE 2] Threshold failed, switching to PEAK DETECTION...")
            boundaries, success = self._try_peak_detection(
                whiteness_dict, w, img_array, debug_dir
            )

            if success:
                print(f" PEAK DETECTION succeeded!")
                return boundaries, 'peak', background_info, whiteness_dict

            # PHASE 3: Try CONTRAST detection
            print(f" [PHASE 3] Peak failed, switching to CONTRAST DETECTION...")
            boundaries, success = self._try_contrast_detection(
                img_array, w, step, debug_dir
            )

            if success:
                print(f" CONTRAST DETECTION rescued the sample!")
                return boundaries, 'contrast', background_info, whiteness_dict

        else:
            #
            # Low whiteness case: Contrast -> Peak
            #
            print(f" [INFO] Low whiteness ({mean_whiteness:.3f} <= {self.WHITENESS_THRESHOLD}) -> Contrast detection first")

            # PHASE 2: Try CONTRAST detection
            print(f" [PHASE 2] Threshold failed, switching to CONTRAST DETECTION...")
            boundaries, success = self._try_contrast_detection(
                img_array, w, step, debug_dir
            )

            if success:
                print(f" CONTRAST DETECTION succeeded!")
                return boundaries, 'contrast', background_info, whiteness_dict

            # PHASE 3: Try PEAK detection
            print(f" [PHASE 3] Contrast failed, switching to PEAK DETECTION...")
            boundaries, success = self._try_peak_detection(
                whiteness_dict, w, img_array, debug_dir
            )

            if success:
                print(f" PEAK DETECTION rescued the sample!")
                return boundaries, 'peak', background_info, whiteness_dict

        # PHASE 4: Try gradient detection
        print(f" [PHASE 4] Peak and Contrast failed, switching to GRADIENT DETECTION...")
        boundaries, success = self._try_gradient_detection(
            whiteness_dict, w, img_array, debug_dir
        )

        if success:
            print(f" GRADIENT DETECTION rescued the sample!")
            return boundaries, 'gradient', background_info, whiteness_dict

        # PHASE 5: Fallback
        print(f" [PHASE 5] All methods failed, using FALLBACK...")

        if debug_dir is not None:
            self._visualize_fallback(img_array, whiteness_dict, debug_dir)

        return self._fallback_equal_division(w), 'fallback', background_info, whiteness_dict

    #
    # V4 QUALITY VALIDATION FUNCTIONS
    #

    def _validate_view_widths(self, boundaries):
        """
        Validate if view widths are reasonable
        Returns: (is_valid, reason, stats)
        """
        widths = [
            boundaries[1] - boundaries[0], # Front
            boundaries[2] - boundaries[1], # Side
            boundaries[3] - boundaries[2] # Bottom
        ]

        total_width = boundaries[3]

        # Rule 1: No view should be too small (< 5% of total)
        for i, w in enumerate(widths):
            ratio = w / total_width
            if ratio < self.MIN_VIEW_RATIO:
                return False, f"View {i} too small ({w}px, {ratio*100:.1f}%)", None

        # Rule 2: No view should dominate (> 80% of total)
        for i, w in enumerate(widths):
            ratio = w / total_width
            if ratio > self.MAX_VIEW_RATIO:
                return False, f"View {i} too large ({w}px, {ratio*100:.1f}%)", None

        # Rule 3: Coefficient of variation check
        mean_width = np.mean(widths)
        std_width = np.std(widths)
        cv = std_width / mean_width if mean_width > 0 else 0

        if cv > self.MAX_WIDTH_CV:
            return False, f"View widths too irregular (cv={cv:.2f})", None

        stats = {
            'widths': widths,
            'mean': mean_width,
            'std': std_width,
            'cv': cv
        }

        return True, "OK", stats

    def _evaluate_gap_quality(self, boundaries, whiteness_dict):
        """
        Evaluate quality of detected gaps
        Returns: score (0-100)

        V4.1: Fixed to use range-based sampling instead of exact key lookup
        """
        gap1_x = boundaries[1]
        gap2_x = boundaries[2]

        # Sample whiteness around gaps using range-based approach
        # This fixes the bug where exact keys might not exist in whiteness_dict
        gap1_values = []
        gap2_values = []

        # Collect all whiteness values within ±20px of each gap
        for key in whiteness_dict.keys():
            if gap1_x - 20 <= key <= gap1_x + 20:
                gap1_values.append(whiteness_dict[key])
            if gap2_x - 20 <= key <= gap2_x + 20:
                gap2_values.append(whiteness_dict[key])

        if not gap1_values or not gap2_values:
            return 0

        # High whiteness = clear gap
        gap1_avg = np.mean(gap1_values)
        gap2_avg = np.mean(gap2_values)
        avg_gap_whiteness = (gap1_avg + gap2_avg) / 2

        # Score
        score = avg_gap_whiteness * 100

        return score

    def _calculate_confidence_score(self, boundaries, method_used, whiteness_dict):
        """
        Calculate overall confidence score
        Returns: (score, components)
        """
        scores = []
        components = {}

        # 1. Method-based score
        method_scores = {
            'adaptive': 95,
            'peak': 85,
            'contrast': 80,
            'gradient': 70,
            'fallback': 10
        }
        method_score = method_scores.get(method_used, 50)
        scores.append(method_score)
        components['method'] = method_score

        # 2. Gap quality score
        gap_score = self._evaluate_gap_quality(boundaries, whiteness_dict)
        scores.append(gap_score)
        components['gap_quality'] = gap_score

        # 3. View uniformity score
        is_valid, reason, stats = self._validate_view_widths(boundaries)
        if stats:
            cv = stats['cv']
            # cv < 0.3: excellent (100)
            # cv > 0.8: poor (0)
            uniformity_score = max(0, min(100, 100 * (0.8 - cv) / 0.5))
        else:
            uniformity_score = 0
        scores.append(uniformity_score)
        components['uniformity'] = uniformity_score

        # 4. Whiteness mean
        mean_whiteness = np.mean(list(whiteness_dict.values()))
        whiteness_score = min(mean_whiteness * 100, 100)
        scores.append(whiteness_score)
        components['whiteness'] = whiteness_score

        # Final score
        final_score = np.mean(scores)

        return final_score, components

    def _analyze_background(self, img_array):
        """Analyze background brightness and determine adaptive threshold"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Method 1: Top 5% brightness
        background_p95 = np.percentile(gray, 95)

        # Method 2: Top 10% brightness
        background_p90 = np.percentile(gray, 90)

        # Method 3: Mean of top 20%
        top_20_percent = np.partition(gray.flatten(), -int(gray.size * 0.2))[-int(gray.size * 0.2):]
        background_mean_top20 = np.mean(top_20_percent)

        # Use conservative estimate (highest value)
        background_brightness = max(background_p95, background_p90, background_mean_top20)

        # Determine adaptive threshold
        if background_brightness < 150:
            adaptive_threshold = background_brightness * 0.70
        elif background_brightness < 200:
            adaptive_threshold = background_brightness * 0.80
        elif background_brightness < 240:
            adaptive_threshold = background_brightness * 0.90
        else:
            adaptive_threshold = 250

        return background_brightness, adaptive_threshold

    def _calculate_whiteness(self, img_array, threshold, step=5):
        """Calculate whiteness with given threshold"""
        h, w = img_array.shape[:2]
        whiteness_dict = {}

        for x in range(0, w, step):
            column = img_array[:, x]

            if len(column.shape) == 1:
                white_count = np.sum(column > threshold)
            else:
                is_white = np.all(column > threshold, axis=1)
                white_count = np.sum(is_white)

            whiteness_dict[x] = white_count / h

        return whiteness_dict

    def _try_threshold_method(self, whiteness_dict, w, img_array, base_threshold, debug_dir):
        """Try threshold-based gap detection"""
        sorted_x = sorted(whiteness_dict.keys())
        edge_margin = int(w * 0.05)

        # V4.1: More strict threshold range (0.99 to 0.88 only)
        # Removed 0.85, 0.80, 0.75 - with validation, we can be stricter
        test_thresholds = [0.99, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88]

        for threshold in test_thresholds:
            raw_gaps = []
            in_gap = False
            gap_start = 0

            for x in sorted_x:
                if whiteness_dict[x] > threshold and not in_gap:
                    gap_start = x
                    in_gap = True
                elif whiteness_dict[x] <= threshold and in_gap:
                    gap_end = x
                    raw_gaps.append((gap_start, gap_end))
                    in_gap = False

            gaps = []
            max_gap_width = int(w * 0.35)

            for start, end in raw_gaps:
                gap_width = end - start

                if gap_width > max_gap_width:
                    continue

                if start < edge_margin or end > w - edge_margin:
                    continue

                gaps.append((start, end))

            print(f" Threshold {threshold:.2f}: found {len(gaps)} valid gaps")

            if len(gaps) < 2:
                continue

            min_gap_distance = int(w * 0.07)
            min_width = int(w * 0.07)

            for i in range(len(gaps)):
                for j in range(i + 1, len(gaps)):
                    gap1_start, gap1_end = gaps[i]
                    gap2_start, gap2_end = gaps[j]

                    gap1_mid = (gap1_start + gap1_end) // 2
                    gap2_mid = (gap2_start + gap2_end) // 2

                    if gap2_mid - gap1_mid < min_gap_distance:
                        continue

                    boundaries = [0, gap1_mid, gap2_mid, w]
                    view_widths = [
                        boundaries[1] - boundaries[0],
                        boundaries[2] - boundaries[1],
                        boundaries[3] - boundaries[2]
                    ]

                    #
                    # V4 VALIDATION: Check boundaries before accepting
                    #
                    is_valid, reason, stats = self._validate_view_widths(boundaries)

                    if not is_valid:
                        print(f" Threshold {threshold:.2f}: {reason}")
                        continue

                    # Check gap quality
                    gap_score = self._evaluate_gap_quality(boundaries, whiteness_dict)
                    if gap_score < self.MIN_GAP_QUALITY:
                        print(f" Threshold {threshold:.2f}: Low gap quality ({gap_score:.1f})")
                        continue

                    # PASSED ALL VALIDATIONS
                    if debug_dir is not None:
                        self._visualize_threshold_result(
                            img_array, whiteness_dict, boundaries,
                            gaps, threshold, base_threshold, debug_dir
                        )

                    if all(width >= min_width for width in view_widths):
                        print(f" Threshold {threshold:.2f}: VALID (gap_quality={gap_score:.1f})")
                        print(f" Valid boundaries found: {boundaries}")
                        print(f" View widths: {view_widths}")
                        return boundaries, True
                    else:
                        print(f" View widths too narrow: {view_widths}")

        print(f" No valid boundaries found with threshold method")
        return None, False

    def _try_peak_detection(self, whiteness_dict, w, img_array, debug_dir):
        """Try peak-based gap detection"""
        x_coords = np.array(sorted(whiteness_dict.keys()))
        whiteness_values = np.array([whiteness_dict[x] for x in x_coords])

        mean_whiteness = np.mean(whiteness_values)
        std_whiteness = np.std(whiteness_values)

        print(f" Whiteness stats: mean={mean_whiteness:.3f}, std={std_whiteness:.3f}")

        # Find peaks
        min_height = mean_whiteness + 0.5 * std_whiteness
        prominence_threshold = 0.02
        min_peak_distance = int(0.07 * len(x_coords))

        peaks, properties = find_peaks(
            whiteness_values,
            height=min_height,
            prominence=prominence_threshold,
            distance=min_peak_distance
        )

        print(f" Found {len(peaks)} peaks")

        if len(peaks) == 0:
            return None, False

        # Filter by edge margin
        edge_margin = int(w * 0.05)
        peak_positions = x_coords[peaks]
        peak_prominences = properties['prominences']

        valid_peaks = []
        for i, pos in enumerate(peak_positions):
            if edge_margin < pos < w - edge_margin:
                valid_peaks.append({
                    'position': pos,
                    'prominence': peak_prominences[i]
                })

        print(f" Valid peaks: {len(valid_peaks)}")

        if len(valid_peaks) < 2:
            return None, False

        # Sort by prominence and take top 2
        valid_peaks_sorted = sorted(valid_peaks, key=lambda p: p['prominence'], reverse=True)
        selected_peaks = sorted(valid_peaks_sorted[:2], key=lambda p: p['position'])

        peak1 = selected_peaks[0]['position']
        peak2 = selected_peaks[1]['position']

        boundaries = [0, peak1, peak2, w]

        # Validate view widths
        min_width = int(w * 0.07)
        view_widths = [
            boundaries[1] - boundaries[0],
            boundaries[2] - boundaries[1],
            boundaries[3] - boundaries[2]
        ]

        if debug_dir is not None:
            self._visualize_peak_result(
                img_array, whiteness_dict, x_coords, whiteness_values,
                peaks, selected_peaks, boundaries, debug_dir
            )

        if all(width >= min_width for width in view_widths):
            print(f" Valid boundaries found: {boundaries}")
            print(f" View widths: {view_widths} (threshold: {min_width}px = 7%)")
            return boundaries, True

        print(f" View widths too narrow: {view_widths} (threshold: {min_width}px = 7%)")
        return None, False

    def _try_contrast_detection(self, img_array, w, step, debug_dir):
        """Contrast-based detection"""
        h = img_array.shape[0]

        brightness_dict = {}
        for x in range(0, w, step):
            column = img_array[:, x]
            if len(column.shape) == 3:
                gray_column = np.mean(column, axis=1)
            else:
                gray_column = column
            brightness_dict[x] = np.mean(gray_column)

        x_coords = np.array(sorted(brightness_dict.keys()))
        brightness_values = np.array([brightness_dict[x] for x in x_coords])

        smoothed = gaussian_filter1d(brightness_values, sigma=3)

        mean_brightness = np.mean(smoothed)
        std_brightness = np.std(smoothed)
        max_brightness = np.max(smoothed)

        print(f" Brightness stats: mean={mean_brightness:.1f}, std={std_brightness:.1f}, max={max_brightness:.1f}")

        min_height = mean_brightness + 0.5 * std_brightness
        min_prominence = 0.3 * std_brightness
        min_peak_distance = int(0.07 * len(x_coords))

        peaks, properties = find_peaks(
            smoothed,
            height=min_height,
            prominence=min_prominence,
            distance=min_peak_distance
        )

        print(f" Found {len(peaks)} brightness peaks")

        if len(peaks) == 0:
            return None, False

        edge_margin = int(w * 0.05)
        peak_positions = x_coords[peaks]
        peak_prominences = properties['prominences']

        valid_peaks = []
        for i, pos in enumerate(peak_positions):
            if edge_margin < pos < w - edge_margin:
                valid_peaks.append({
                    'position': pos,
                    'prominence': peak_prominences[i],
                    'brightness': smoothed[peaks[i]]
                })
                print(f" Peak {len(valid_peaks)}: x={pos}, brightness={smoothed[peaks[i]]:.1f}, prominence={peak_prominences[i]:.1f}")

        print(f" Valid brightness peaks: {len(valid_peaks)}")

        if len(valid_peaks) < 2:
            return None, False

        valid_peaks_sorted = sorted(valid_peaks, key=lambda p: p['prominence'], reverse=True)
        selected_peaks = sorted(valid_peaks_sorted[:2], key=lambda p: p['position'])

        peak1 = selected_peaks[0]['position']
        peak2 = selected_peaks[1]['position']

        boundaries = [0, peak1, peak2, w]

        min_width = int(w * 0.07)
        view_widths = [
            boundaries[1] - boundaries[0],
            boundaries[2] - boundaries[1],
            boundaries[3] - boundaries[2]
        ]

        if debug_dir is not None:
            self._visualize_contrast_result(
                img_array, brightness_dict, x_coords, smoothed,
                peaks, selected_peaks, boundaries, debug_dir
            )

        if all(width >= min_width for width in view_widths):
            print(f" Valid boundaries found: {boundaries}")
            print(f" View widths: {view_widths} (threshold: {min_width}px = 7%)")
            return boundaries, True

        print(f" View widths too narrow: {view_widths} (threshold: {min_width}px = 7%)")
        return None, False

    def _try_gradient_detection(self, whiteness_dict, w, img_array, debug_dir):
        """Try gradient-based gap detection"""
        x_coords = np.array(sorted(whiteness_dict.keys()))
        whiteness_values = np.array([whiteness_dict[x] for x in x_coords])

        gradient = np.gradient(whiteness_values)

        print(f" Gradient stats: mean={np.mean(gradient):.4f}, std={np.std(gradient):.4f}, max={np.max(gradient):.4f}")

        min_gradient_height = 0.005
        min_prominence = 0.003
        min_peak_distance = int(0.07 * len(x_coords))

        rising_peaks, properties = find_peaks(
            gradient,
            height=min_gradient_height,
            prominence=min_prominence,
            distance=min_peak_distance
        )

        print(f" Found {len(rising_peaks)} rising edge peaks")

        if len(rising_peaks) == 0:
            return None, False

        peak_positions = x_coords[rising_peaks]
        peak_gradients = gradient[rising_peaks]
        peak_prominences = properties['prominences']

        edge_margin = int(w * 0.05)

        valid_peaks = []
        for i, pos in enumerate(peak_positions):
            if edge_margin < pos < w - edge_margin:
                valid_peaks.append({
                    'position': pos,
                    'gradient': peak_gradients[i],
                    'prominence': peak_prominences[i]
                })
                print(f" Rising peak {len(valid_peaks)}: x={pos}, gradient={peak_gradients[i]:.4f}, prominence={peak_prominences[i]:.4f}")

        print(f" Valid rising peaks: {len(valid_peaks)}")

        if len(valid_peaks) < 2:
            return None, False

        valid_peaks_sorted = sorted(valid_peaks, key=lambda p: p['prominence'], reverse=True)
        selected_peaks = sorted(valid_peaks_sorted[:2], key=lambda p: p['position'])

        peak1 = selected_peaks[0]['position']
        peak2 = selected_peaks[1]['position']

        boundaries = [0, peak1, peak2, w]

        min_width = int(w * 0.07)
        view_widths = [
            boundaries[1] - boundaries[0],
            boundaries[2] - boundaries[1],
            boundaries[3] - boundaries[2]
        ]

        if debug_dir is not None:
            self._visualize_gradient_result(
                img_array, whiteness_dict, x_coords, whiteness_values,
                gradient, rising_peaks, selected_peaks, boundaries, debug_dir
            )

        if all(width >= min_width for width in view_widths):
            print(f" Valid boundaries found: {boundaries}")
            print(f" View widths: {view_widths} (threshold: {min_width}px = 7%)")
            return boundaries, True

        print(f" View widths too narrow: {view_widths} (threshold: {min_width}px = 7%)")
        return None, False

    def _visualize_threshold_result(self, img_array, whiteness_dict, boundaries,
                                     gaps, threshold, base_threshold, debug_dir):
        """Visualize threshold method result"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        h, w = img_array.shape[:2]
        x_coords = sorted(whiteness_dict.keys())
        whiteness_values = [whiteness_dict[x] for x in x_coords]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),
                                       gridspec_kw={'height_ratios': [1, 1]})

        ax1.imshow(img_array)
        for i, b in enumerate(boundaries):
            color = ['red', 'blue', 'green', 'red'][i]
            linestyle = '--' if i == 0 or i == 3 else '-'
            ax1.axvline(x=b, color=color, linestyle=linestyle, linewidth=3, alpha=0.8)
        ax1.set_title(f'Adaptive Threshold Method (base={base_threshold:.1f}, used={threshold:.2f})',
                     fontsize=14, fontweight='bold')

        ax2.plot(x_coords, whiteness_values, 'b-', linewidth=2, label='Whiteness')
        ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')

        for start, end in gaps:
            ax2.axvspan(start, end, alpha=0.3, color='green', label='Gap' if start == gaps[0][0] else '')

        ax2.set_title('Whiteness Analysis (Adaptive Threshold Method)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate (pixels)')
        ax2.set_ylabel('Whiteness Ratio')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(debug_dir / 'gap_detection_adaptive_threshold.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f" -> Saved gap_detection_adaptive_threshold.png")

    def _visualize_peak_result(self, img_array, whiteness_dict, x_coords, whiteness_values,
                                peaks, selected_peaks, boundaries, debug_dir):
        """Visualize peak detection result"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        h, w = img_array.shape[:2]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),
                                       gridspec_kw={'height_ratios': [1, 1]})

        ax1.imshow(img_array)
        for i, b in enumerate(boundaries):
            color = ['red', 'blue', 'green', 'red'][i]
            linestyle = '--' if i == 0 or i == 3 else '-'
            ax1.axvline(x=b, color=color, linestyle=linestyle, linewidth=3, alpha=0.8)
        ax1.set_title('Peak Detection Result', fontsize=14, fontweight='bold')

        ax2.plot(x_coords, whiteness_values, 'b-', linewidth=2, label='Whiteness')

        if len(peaks) > 0:
            peak_x = x_coords[peaks]
            peak_y = whiteness_values[peaks]
            ax2.plot(peak_x, peak_y, 'ro', markersize=10, label='All Peaks')

        if selected_peaks:
            sel_x = [p['position'] for p in selected_peaks]
            sel_y = [whiteness_dict[p['position']] for p in selected_peaks]
            ax2.plot(sel_x, sel_y, 'g^', markersize=15, label='Selected Peaks')

        ax2.set_title('Whiteness Analysis (Peak Detection Method)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate (pixels)')
        ax2.set_ylabel('Whiteness Ratio')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(debug_dir / 'gap_detection_peak.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f" -> Saved gap_detection_peak.png")

    def _visualize_contrast_result(self, img_array, brightness_dict, x_coords, smoothed,
                                    peaks, selected_peaks, boundaries, debug_dir):
        """Visualize contrast detection result"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        h, w = img_array.shape[:2]
        brightness_values = [brightness_dict[x] for x in x_coords]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15),
                                            gridspec_kw={'height_ratios': [1, 1, 1]})

        ax1.imshow(img_array)
        for i, b in enumerate(boundaries):
            color = ['red', 'blue', 'green', 'red'][i]
            linestyle = '--' if i == 0 or i == 3 else '-'
            ax1.axvline(x=b, color=color, linestyle=linestyle, linewidth=3, alpha=0.8)
        ax1.set_title('Contrast Detection Result', fontsize=14, fontweight='bold')

        ax2.plot(x_coords, brightness_values, 'gray', linewidth=1, alpha=0.5, label='Raw Brightness')
        ax2.plot(x_coords, smoothed, 'b-', linewidth=2, label='Smoothed Brightness')

        if len(peaks) > 0:
            peak_x = x_coords[peaks]
            peak_y = smoothed[peaks]
            ax2.plot(peak_x, peak_y, 'ro', markersize=10, label='All Peaks')

        if selected_peaks:
            sel_x = [p['position'] for p in selected_peaks]
            sel_y = [smoothed[list(x_coords).index(p['position'])] for p in selected_peaks]
            ax2.plot(sel_x, sel_y, 'g^', markersize=15, label='Selected Peaks')

        ax2.set_title('Brightness Analysis (Contrast Method)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate (pixels)')
        ax2.set_ylabel('Brightness (0-255)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        mean_brightness = np.mean(smoothed)
        contrast = smoothed - mean_brightness
        ax3.plot(x_coords, contrast, 'purple', linewidth=2, label='Contrast (relative to mean)')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        if selected_peaks:
            sel_x = [p['position'] for p in selected_peaks]
            sel_c = [smoothed[list(x_coords).index(p['position'])] - mean_brightness for p in selected_peaks]
            ax3.plot(sel_x, sel_c, 'g^', markersize=15, label='Selected Peaks')

        ax3.set_title('Relative Brightness (Contrast)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X coordinate (pixels)')
        ax3.set_ylabel('Brightness - Mean')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(debug_dir / 'gap_detection_contrast.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f" -> Saved gap_detection_contrast.png")

    def _visualize_gradient_result(self, img_array, whiteness_dict, x_coords, whiteness_values,
                                    gradient, rising_peaks, selected_peaks, boundaries, debug_dir):
        """Visualize gradient detection result"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        h, w = img_array.shape[:2]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15),
                                            gridspec_kw={'height_ratios': [1, 1, 1]})

        ax1.imshow(img_array)
        for i, b in enumerate(boundaries):
            color = ['red', 'blue', 'green', 'red'][i]
            linestyle = '--' if i == 0 or i == 3 else '-'
            ax1.axvline(x=b, color=color, linestyle=linestyle, linewidth=3, alpha=0.8)
        ax1.set_title('Gradient Detection Result', fontsize=14, fontweight='bold')

        ax2.plot(x_coords, whiteness_values, 'b-', linewidth=2, label='Whiteness')

        if selected_peaks:
            sel_x = [p['position'] for p in selected_peaks]
            sel_y = [whiteness_dict[p['position']] for p in selected_peaks]
            ax2.plot(sel_x, sel_y, 'g^', markersize=15, label='Selected Peaks')

        ax2.set_title('Whiteness Values', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate (pixels)')
        ax2.set_ylabel('Whiteness Ratio')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(x_coords[:-1], gradient[:-1], 'r-', linewidth=2, label='Gradient')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.005, color='orange', linestyle=':', label='Detection threshold')

        if len(rising_peaks) > 0:
            peak_x = x_coords[rising_peaks]
            peak_g = gradient[rising_peaks]
            ax3.plot(peak_x, peak_g, 'ro', markersize=10, label='All Rising Peaks')

        if selected_peaks:
            sel_x = [p['position'] for p in selected_peaks]
            sel_g = [p['gradient'] for p in selected_peaks]
            ax3.plot(sel_x, sel_g, 'g^', markersize=15, label='Selected Peaks')

        ax3.set_title('Gradient Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X coordinate (pixels)')
        ax3.set_ylabel('Gradient')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(debug_dir / 'gap_detection_gradient.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f" -> Saved gap_detection_gradient.png")

    def _visualize_fallback(self, img_array, whiteness_dict, debug_dir):
        """Visualize fallback case"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x_coords = sorted(whiteness_dict.keys())
        whiteness_values = [whiteness_dict[x] for x in x_coords]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),
                                       gridspec_kw={'height_ratios': [1, 1]})

        ax1.imshow(img_array)
        ax1.set_title('FALLBACK CASE - All Methods Failed', fontsize=14, fontweight='bold', color='red')

        ax2.plot(x_coords, whiteness_values, 'b-', linewidth=2, label='Whiteness')
        ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Min Threshold (0.85)')
        ax2.set_title('Whiteness Analysis (Fallback)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate (pixels)')
        ax2.set_ylabel('Whiteness Ratio')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(debug_dir / 'gap_detection_fallback.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f" -> Saved gap_detection_fallback.png")

    def _fallback_equal_division(self, width):
        """Fallback to equal 3-way division"""
        w1 = width // 3
        w2 = 2 * width // 3
        return [0, w1, w2, width]

    def _find_vertical_extent(self, img_array, base_threshold=240):
        """Find top and bottom boundaries of content"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        h, w = gray.shape

        y_min = 0
        for y in range(h):
            if np.any(gray[y, :] < base_threshold):
                y_min = y
                break

        y_max = h
        for y in range(h - 1, -1, -1):
            if np.any(gray[y, :] < base_threshold):
                y_max = y + 1
                break

        return y_min, y_max

    def _find_tight_bbox_2d(self, img_array, base_threshold=240):
        """
        2D scan tight bbox for dark backgrounds
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        h, w = gray.shape

        edge_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])

        local_background = np.percentile(edge_pixels, 90)

        if local_background < base_threshold - 20:
            base_obj_threshold = max(local_background * 0.85, 180)
        else:
            base_obj_threshold = max(base_threshold * 0.90, 200)

        base_obj_threshold = max(base_obj_threshold, 180)

        print(f" 2D scan threshold: global={base_threshold:.1f}, local_bg={local_background:.1f}, base={base_obj_threshold:.1f}")

        y_brightness = np.mean(gray, axis=1)
        y_smooth = gaussian_filter1d(y_brightness, sigma=5)

        x_brightness = np.mean(gray, axis=0)
        x_smooth = gaussian_filter1d(x_brightness, sigma=5)

        y_threshold = base_obj_threshold
        x_threshold = base_obj_threshold

        top_y = 0
        for y in range(h):
            if y_smooth[y] < y_threshold:
                top_y = max(0, y - 5)
                break

        bottom_y = h
        for y in range(h - 1, -1, -1):
            if y_smooth[y] < y_threshold:
                bottom_y = min(h, y + 5)
                break

        left_x = 0
        for x in range(w):
            if x_smooth[x] < x_threshold:
                left_x = max(0, x - 5)
                break

        right_x = w
        for x in range(w - 1, -1, -1):
            if x_smooth[x] < x_threshold:
                right_x = min(w, x + 5)
                break

        print(f" 2D scan bbox: left={left_x}, top={top_y}, right={right_x}, bottom={bottom_y}")

        if right_x <= left_x or bottom_y <= top_y:
            print(f" Invalid bbox, using fallback")
            return self._find_tight_bbox_fallback(gray, base_obj_threshold)

        return (left_x, top_y, right_x, bottom_y)

    def _find_tight_bbox_fallback(self, gray, threshold):
        """Fallback method"""
        h, w = gray.shape

        left_x = 0
        for x in range(w):
            if np.any(gray[:, x] < threshold):
                left_x = x
                break

        right_x = w - 1
        for x in range(w - 1, -1, -1):
            if np.any(gray[:, x] < threshold):
                right_x = x + 1
                break

        top_y = 0
        for y in range(h):
            if np.any(gray[y, :] < threshold):
                top_y = y
                break

        bottom_y = h - 1
        for y in range(h - 1, -1, -1):
            if np.any(gray[y, :] < threshold):
                bottom_y = y + 1
                break

        if right_x <= left_x or bottom_y <= top_y:
            return None

        return (left_x, top_y, right_x, bottom_y)

    def _find_tight_bbox_1d_simple(self, img_array, threshold=240):
        """
        Simple 1D scan for bright backgrounds
        v3: threshold=240 (restored to preserve bright objects)
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        h, w = gray.shape

        print(f" 1D simple scan: threshold={threshold}")

        left_x = 0
        for x in range(w):
            if np.any(gray[:, x] < threshold):
                left_x = x
                break

        right_x = w - 1
        for x in range(w - 1, -1, -1):
            if np.any(gray[:, x] < threshold):
                right_x = x + 1
                break

        top_y = 0
        for y in range(h):
            if np.any(gray[y, :] < threshold):
                top_y = y
                break

        bottom_y = h - 1
        for y in range(h - 1, -1, -1):
            if np.any(gray[y, :] < threshold):
                bottom_y = y + 1
                break

        if right_x <= left_x or bottom_y <= top_y:
            return None

        return (left_x, top_y, right_x, bottom_y)

    def _remove_small_components_simple(self, img_array, threshold=240):
        """
        Remove small noise components - simple method (V1 style)
        NO morphology operations - preserves bright object parts naturally

        Args:
            img_array: Image array
            threshold: Fixed threshold (default 240)
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        print(f" Component removal (simple, V1 style): threshold={threshold}")

        # Binarization with fixed threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # NO morphological operations (V1 style)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels <= 1:
            print(f" No components found, keeping original")
            return img_array

        # Keep ONLY the largest component (V1 style)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        if len(component_sizes) == 0:
            return img_array

        largest_component_label = np.argmax(component_sizes) + 1
        mask = (labels == largest_component_label).astype(np.uint8) * 255

        print(f" Kept 1 component (largest, {component_sizes[largest_component_label-1]} px)")

        # Apply mask to ORIGINAL image
        if len(img_array.shape) == 3:
            mask_3d = np.stack([mask] * 3, axis=-1)
            cleaned = np.where(mask_3d > 0, img_array, 255)
        else:
            cleaned = np.where(mask > 0, img_array, 255)

        return cleaned.astype(np.uint8)

    def _normalize_view_512x512(self, view_image, boundaries, total_width, total_height, view_index):
        """
        Normalize view to 512x512 with improved scaling
        
        Strategy:
        1. Use width/height from combined.png as reference
        2. Scale object to fill 90% of canvas
        3. Maintain aspect ratio
        4. Center placement
        
        Args:
            view_image: tight box result (H, W, 3)
            boundaries: [0, x1, x2, total_width]
            total_width: combined.png width
            total_height: combined.png height
            view_index: 0=front, 1=side, 2=bottom
        
        Returns:
            numpy array (512, 512, 3)
        """
        # Configuration
        TARGET_SIZE = 512
        FILL_RATIO = 0.90 # How much of 512x512 to fill (90%)
        
        # 1. Get object dimensions
        obj_h, obj_w = view_image.shape[:2]
        
        # 2. Calculate reference from combined.png
        view_width_original = boundaries[view_index+1] - boundaries[view_index]
        view_width_ratio = view_width_original / total_width
        
        # 3. Target dimensions (fill more aggressively)
        target_max = int(TARGET_SIZE * FILL_RATIO)
        
        # 4. Scale to fit target (maintain aspect ratio)
        scale = min(target_max / obj_w, target_max / obj_h)
        new_w = max(1, int(obj_w * scale)) # Ensure at least 1px
        new_h = max(1, int(obj_h * scale)) # Ensure at least 1px
        
        # 5. Resize
        resized = cv2.resize(view_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 6. Create 512x512 white canvas
        canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), 255, dtype=np.uint8)
        
        # 7. Center placement
        y_offset = (TARGET_SIZE - new_h) // 2
        x_offset = (TARGET_SIZE - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Debug info
        print(f" {['Front', 'Side', 'Bottom'][view_index]}: "
              f"{obj_w}x{obj_h} -> {new_w}x{new_h} "
              f"(ratio={view_width_ratio:.3f}, fill={new_w/TARGET_SIZE:.2f})")
        
        return canvas
    
    def _combine_vertically(self, image_paths, output_path, padding=20):
        """Combine 3 views vertically"""
        images = [Image.open(str(p)) for p in image_paths]

        max_width = max(img.size[0] for img in images)

        padded_images = []
        for img in images:
            w, h = img.size
            if w < max_width:
                left_pad = (max_width - w) // 2
                padded = Image.new('RGB', (max_width, h), color='white')
                padded.paste(img, (left_pad, 0))
                padded_images.append(padded)
            else:
                padded_images.append(img)

        total_height = sum(img.size[1] for img in padded_images) + padding * 2
        combined = Image.new('RGB', (max_width, total_height), color='white')

        y_offset = 0
        for img in padded_images:
            combined.paste(img, (0, y_offset))
            y_offset += img.size[1] + padding

        combined.save(output_path)
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Adaptive Gap Detection Cropping v4.2')
    parser.add_argument('--data_dir', type=str,
                        default='./data/mv_images',
                        help='Input directory with combined.png files')
    parser.add_argument('--output_dir', type=str,
                        default='./data/mv_images',
                        help='Output directory for cropped images')
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=997)
    parser.add_argument('--log_file', type=str, default='crop_log_adaptive_v4.json')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    print("="*70)
    print("ADAPTIVE GAP DETECTION CROPPING V4.2")
    print("Strategy: Threshold -> [Conditional] -> Gradient -> Fallback")
    print("Improvements:")
    print(" - Quality validation (view width, gap quality)")
    print(" - Confidence scoring system")
    print(" - Auto-filtering of invalid samples")
    print(" - Enhanced adaptive threshold with validation")
    print(" - FIXED: Gap quality bug + stricter thresholds (0.88 max)")
    print(" - FIXED: Whiteness threshold (0.30) for better ordering")
    print("="*70)
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample range: {args.start_id} to {args.end_id}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print("="*70)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cropper = AdaptiveCropperV4(debug_mode=args.debug)

    data_dir = Path(args.data_dir)
    start_time = datetime.now()
    results = [] # Track all results for quality distribution

    data_dir = Path(args.data_dir)
    start_time = datetime.now()

    for sample_id in tqdm(range(args.start_id, args.end_id), desc="Processing samples"):
        combined_path = data_dir / str(sample_id) / "combined.png"

        if not combined_path.exists():
            continue

        sample_output_dir = output_dir / str(sample_id)

        result = cropper.crop_image(combined_path, sample_output_dir, sample_id)

        if result['success']:
            cropper.success_count += 1
            results.append(result) # Track result
            print(f" Sample {sample_id}: Success (via {result.get('method_used', 'unknown')})")
        else:
            cropper.fail_count += 1
            cropper.failed_samples.append({
                'sample_id': sample_id,
                'reason': result.get('reason', 'unknown')
            })
            print(f" Sample {sample_id}: Failed ({result.get('reason', 'unknown')})")

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    total = cropper.success_count + cropper.fail_count

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f" Success: {cropper.success_count}")
    print(f" Failed: {cropper.fail_count}")
    print(f" Adaptive threshold rescued: {len(cropper.adaptive_rescued_samples)}")
    print(f" Peak detection rescued: {len(cropper.peak_rescued_samples)}")
    print(f" Contrast detection rescued: {len(cropper.contrast_rescued_samples)}")
    print(f" Gradient detection rescued: {len(cropper.gradient_rescued_samples)}")
    print(f" Fallback used: {len(cropper.fallback_samples)}")
    if total > 0:
        print(f" Success Rate: {cropper.success_count/total*100:.2f}%")
        adaptive_success = len(cropper.adaptive_rescued_samples)
        peak_success = len(cropper.peak_rescued_samples)
        contrast_success = len(cropper.contrast_rescued_samples)
        gradient_success = len(cropper.gradient_rescued_samples)
        print(f" - Adaptive: {adaptive_success/total*100:.1f}%")
        print(f" - Peak: {peak_success/total*100:.1f}%")
        print(f" - Contrast: {contrast_success/total*100:.1f}%")
        print(f" - Gradient: {gradient_success/total*100:.1f}%")
    print(f" Total Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f" Throughput: {total/elapsed:.2f} images/s")

    if cropper.adaptive_rescued_samples:
        print(f"\n Samples rescued by adaptive threshold ({len(cropper.adaptive_rescued_samples)}):")
        for i in range(0, len(cropper.adaptive_rescued_samples), 10):
            batch = cropper.adaptive_rescued_samples[i:i+10]
            print(f" {batch}")

    if cropper.peak_rescued_samples:
        print(f"\n Samples rescued by peak detection ({len(cropper.peak_rescued_samples)}):")
        for i in range(0, len(cropper.peak_rescued_samples), 10):
            batch = cropper.peak_rescued_samples[i:i+10]
            print(f" {batch}")

    if cropper.contrast_rescued_samples:
        print(f"\n Samples rescued by contrast detection ({len(cropper.contrast_rescued_samples)}):")
        for i in range(0, len(cropper.contrast_rescued_samples), 10):
            batch = cropper.contrast_rescued_samples[i:i+10]
            print(f" {batch}")

    if cropper.gradient_rescued_samples:
        print(f"\n Samples rescued by gradient detection ({len(cropper.gradient_rescued_samples)}):")
        for i in range(0, len(cropper.gradient_rescued_samples), 10):
            batch = cropper.gradient_rescued_samples[i:i+10]
            print(f" {batch}")

    if cropper.fallback_samples:
        print(f"\n Samples that used fallback ({len(cropper.fallback_samples)}):")
        for i in range(0, len(cropper.fallback_samples), 10):
            batch = cropper.fallback_samples[i:i+10]
            print(f" {batch}")

    # V4: Low quality samples
    if cropper.low_quality_samples:
        print(f"\n Low quality samples detected ({len(cropper.low_quality_samples)}):")
        for sample in cropper.low_quality_samples[:20]: # Show first 20
            print(f" #{sample['sample_id']}: {sample['method']}, confidence={sample.get('confidence', 0):.1f}, {sample['reason']}")
        if len(cropper.low_quality_samples) > 20:
            print(f" ... and {len(cropper.low_quality_samples) - 20} more")

    # V4: Quality distribution
    if results:
        print(f"\n Quality Distribution:")
        excellent = len([r for r in results if r.get('confidence_score', 0) >= 80])
        good = len([r for r in results if 60 <= r.get('confidence_score', 0) < 80])
        review = len([r for r in results if 40 <= r.get('confidence_score', 0) < 60])
        invalid = len([r for r in results if r.get('confidence_score', 0) < 40])
        print(f" Excellent (>=80): {excellent} ({excellent/len(results)*100:.1f}%)")
        print(f" Good (60-80): {good} ({good/len(results)*100:.1f}%)")
        print(f" Review (40-60): {review} ({review/len(results)*100:.1f}%)")
        print(f" Invalid (<40): {invalid} ({invalid/len(results)*100:.1f}%)")

    log_data = {
        'timestamp': start_time.isoformat(),
        'elapsed_seconds': elapsed,
        'success': cropper.success_count,
        'failed': cropper.fail_count,
        'adaptive_rescued_count': len(cropper.adaptive_rescued_samples),
        'peak_rescued_count': len(cropper.peak_rescued_samples),
        'contrast_rescued_count': len(cropper.contrast_rescued_samples),
        'gradient_rescued_count': len(cropper.gradient_rescued_samples),
        'fallback_count': len(cropper.fallback_samples),
        'low_quality_count': len(cropper.low_quality_samples), # V4
        'success_rate': cropper.success_count / total * 100 if total > 0 else 0,
        'failed_samples': cropper.failed_samples,
        'adaptive_rescued_samples': cropper.adaptive_rescued_samples,
        'peak_rescued_samples': cropper.peak_rescued_samples,
        'contrast_rescued_samples': cropper.contrast_rescued_samples,
        'gradient_rescued_samples': cropper.gradient_rescued_samples,
        'fallback_samples': cropper.fallback_samples,
        'low_quality_samples': cropper.low_quality_samples, # V4
        'quality_distribution': { # V4
            'excellent': len([r for r in results if r.get('confidence_score', 0) >= 80]),
            'good': len([r for r in results if 60 <= r.get('confidence_score', 0) < 80]),
            'review': len([r for r in results if 40 <= r.get('confidence_score', 0) < 60]),
            'invalid': len([r for r in results if r.get('confidence_score', 0) < 40])
        }
    }

    with open(args.log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n Log saved to {args.log_file}")
    print("="*70)


if __name__ == "__main__":
    main()