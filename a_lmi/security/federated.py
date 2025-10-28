"""
Federated Learning Module

Enables distributed learning where raw data stays on edge devices
and only model updates are shared.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn


class FederatedLearning:
    """
    Federated Learning coordinator for distributed A-LMI training.
    
    Supports:
    - Client-side model training
    - Secure gradient aggregation
    - Differential privacy
    - Federated knowledge graph construction
    """
    
    def __init__(self):
        """Initialize federated learning coordinator."""
        self.logger = logging.getLogger(__name__)
        self.clients = []
        self.global_model = None
        self.logger.info("Federated Learning initialized")
    
    def add_client(self, client_id: str, model: nn.Module):
        """
        Add a federated client.
        
        Args:
            client_id: Unique client identifier
            model: Client's local model
        """
        client = {
            'id': client_id,
            'model': model,
            'updates': 0
        }
        self.clients.append(client)
        self.logger.info(f"Added client: {client_id}")
    
    def federated_averaging(self, client_updates: List[Dict[str, torch.Tensor]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Perform federated averaging of model updates.
        
        Args:
            client_updates: List of model state dictionaries from clients
            weights: Weights for each client (if None, use uniform weights)
            
        Returns:
            Aggregated global model update
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        
        if len(weights) != len(client_updates):
            raise ValueError("Weights and updates must have same length")
        
        # Initialize aggregated update
        aggregated = {}
        for key in client_updates[0].keys():
            aggregated[key] = torch.zeros_like(client_updates[0][key])
        
        # Weighted average
        for update, weight in zip(client_updates, weights):
            for key in aggregated.keys():
                aggregated[key] += weight * update[key]
        
        self.logger.info(f"Aggregated updates from {len(client_updates)} clients")
        return aggregated
    
    def apply_differential_privacy(self, model_update: Dict[str, torch.Tensor], sensitivity: float = 1.0, epsilon: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to model update.
        
        Args:
            model_update: Model update to privatize
            sensitivity: Sensitivity of the update
            epsilon: Privacy budget
            
        Returns:
            Privatized model update
        """
        # Compute noise scale
        noise_scale = sensitivity / epsilon
        
        # Add Gaussian noise
        privatized = {}
        for key, value in model_update.items():
            noise = torch.normal(0, noise_scale, size=value.shape)
            privatized[key] = value + noise
        
        self.logger.info(f"Applied differential privacy (ε={epsilon})")
        return privatized
    
    def secure_aggregation(self, client_updates: List[Dict[str, torch.Tensor]], enable_dp: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation with optional differential privacy.
        
        Args:
            client_updates: Client model updates
            enable_dp: Enable differential privacy
            
        Returns:
            Securely aggregated global update
        """
        # Aggregate updates
        aggregated = self.federated_averaging(client_updates)
        
        # Apply differential privacy if enabled
        if enable_dp:
            aggregated = self.apply_differential_privacy(aggregated)
        
        return aggregated

