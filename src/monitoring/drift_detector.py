from scipy import stats


class DriftDetector:
    """
    Detect concept drift in data distributions
    """
    
    def __init__(self, baseline_threshold: float = 0.05):
        self.baseline_threshold = baseline_threshold
        self.baseline_stats = None
        
    def fit_baseline(self, data: np.ndarray):
        """Fit baseline distribution"""
        self.baseline_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'median': np.median(data, axis=0)
        }
        
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, any]:
        """
        Detect drift using statistical tests
        
        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted")
        
        results = {
            'drift_detected': False,
            'drift_features': [],
            'severity': 'NONE'
        }
        
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)
        
        # Compare distributions
        for i in range(len(current_mean)):
            # Z-test for mean shift
            z_score = abs(current_mean[i] - self.baseline_stats['mean'][i]) / \
                     (self.baseline_stats['std'][i] + 1e-10)
            
            if z_score > 3:  # Significant drift
                results['drift_detected'] = True
                results['drift_features'].append(i)
        
        # Calculate severity
        if len(results['drift_features']) > 0:
            drift_ratio = len(results['drift_features']) / len(current_mean)
            if drift_ratio > 0.3:
                results['severity'] = 'HIGH'
            elif drift_ratio > 0.1:
                results['severity'] = 'MEDIUM'
            else:
                results['severity'] = 'LOW'
        
        return results
