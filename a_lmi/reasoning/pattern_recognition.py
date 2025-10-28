"""
Pattern Recognition Engine

Uses spectral signatures for frequency-based pattern discovery.
Implements the novel approach of analyzing information by its vibrational characteristics.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from scipy.stats import pearsonr

from ..core.light_token import LightToken


class SpectralPatternRecognizer:
    """
    Pattern recognition using spectral signatures.
    
    This is the core innovation: finding patterns in data by analyzing
    the frequency characteristics of semantic embeddings rather than
    just semantic similarity.
    """
    
    def __init__(self, config: dict):
        """
        Initialize pattern recognizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.n_clusters = 10  # Number of spectral clusters
        
        # Trained models
        self.clustering_model = None
        self.classification_model = None
    
    def cluster_by_spectrum(self, tokens: List[LightToken]) -> Dict[int, List[LightToken]]:
        """
        Cluster tokens by their spectral signatures.
        
        This groups information that operates at similar frequencies,
        potentially revealing relationships invisible to semantic similarity.
        
        Args:
            tokens: List of Light Tokens to cluster
            
        Returns:
            Dictionary mapping cluster_id to list of tokens
        """
        # Extract spectral features
        spectra = np.array([token.get_spectral_power() for token in tokens])
        
        # Perform clustering (e.g., KMeans on spectral power)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(spectra)
        
        # Group tokens by cluster
        clusters = {}
        for token, label in zip(tokens, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(token)
        
        self.logger.info(f"Clustered {len(tokens)} tokens into {len(clusters)} spectral clusters")
        
        return clusters
    
    def find_frequency_patterns(
        self,
        query_token: LightToken,
        candidate_tokens: List[LightToken],
        threshold: float = 0.8
    ) -> List[Tuple[LightToken, float]]:
        """
        Find tokens with resonant frequencies.
        
        This implements the resonance-based retrieval concept: finding
        information that vibrates at compatible frequencies.
        
        Args:
            query_token: Token to find resonances for
            candidate_tokens: List of candidates
            threshold: Minimum similarity threshold
            
        Returns:
            List of (token, similarity) tuples
        """
        query_spectrum = query_token.get_spectral_power()
        
        matches = []
        for candidate in candidate_tokens:
            candidate_spectrum = candidate.get_spectral_power()
            
            # Compute spectral correlation
            correlation, _ = pearsonr(query_spectrum, candidate_spectrum)
            
            if correlation >= threshold:
                matches.append((candidate, correlation))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def detect_cross_modal_patterns(
        self,
        text_tokens: List[LightToken],
        image_tokens: List[LightToken],
        audio_tokens: List[LightToken]
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns across different modalities using spectral analysis.
        
        Finds text, image, and audio that share similar frequency characteristics
        despite semantic differences - revealing abstract structural similarities.
        
        Args:
            text_tokens: Text tokens
            image_tokens: Image tokens
            audio_tokens: Audio tokens
            
        Returns:
            List of cross-modal pattern matches
        """
        patterns = []
        
        # Compare modalities pairwise
        for text_token in text_tokens[:10]:  # Sample for efficiency
            text_spectrum = text_token.get_spectral_power()
            
            for image_token in image_tokens[:10]:
                image_spectrum = image_token.get_spectral_power()
                correlation, _ = pearsonr(text_spectrum, image_spectrum)
                
                if correlation > 0.7:  # Strong spectral match
                    patterns.append({
                        'type': 'text-image',
                        'text_token': text_token.token_id,
                        'image_token': image_token.token_id,
                        'spectral_correlation': correlation,
                        'text_content': text_token.content_text[:100],
                        'image_uri': image_token.source_uri
                    })
            
            for audio_token in audio_tokens[:10]:
                audio_spectrum = audio_token.get_spectral_power()
                correlation, _ = pearsonr(text_spectrum, audio_spectrum)
                
                if correlation > 0.7:
                    patterns.append({
                        'type': 'text-audio',
                        'text_token': text_token.token_id,
                        'audio_token': audio_token.token_id,
                        'spectral_correlation': correlation,
                        'text_content': text_token.content_text[:100],
                        'audio_ref': audio_token.source_uri
                    })
        
        self.logger.info(f"Found {len(patterns)} cross-modal spectral patterns")
        
        return patterns


class AnomalyDetector:
    """
    Detect anomalies using spectral analysis.
    
    Finds information that doesn't fit expected frequency patterns,
    potentially indicating novel or contradictory information.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_spectral_outliers(
        self,
        tokens: List[LightToken],
        threshold: float = 2.0
    ) -> List[LightToken]:
        """
        Detect tokens with anomalous spectral signatures.
        
        Args:
            tokens: List of tokens to analyze
            threshold: Standard deviations from mean for outlier detection
            
        Returns:
            List of outlier tokens
        """
        spectra = np.array([token.get_spectral_power() for token in tokens])
        
        # Compute mean and std
        mean_spectrum = np.mean(spectra, axis=0)
        std_spectrum = np.std(spectra, axis=0)
        
        # Find outliers
        outliers = []
        for token, spectrum in zip(tokens, spectra):
            deviation = np.abs(spectrum - mean_spectrum) / (std_spectrum + 1e-10)
            max_deviation = np.max(deviation)
            
            if max_deviation > threshold:
                outliers.append(token)
        
        self.logger.info(f"Detected {len(outliers)} spectral outliers")
        
        return outliers

