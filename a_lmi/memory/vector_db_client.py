"""
Vector Database Client for Milvus

Stores and retrieves Light Tokens by semantic similarity using spectral signatures.
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any
import numpy as np
import logging

from ..core.light_token import LightToken


class VectorDBClient:
    """
    Client for Milvus vector database operations.
    
    Implements similarity search on Light Tokens using:
    - Semantic embeddings (joint_embedding)
    - Spectral signatures for frequency-based retrieval
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Milvus connection and collection.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['infrastructure']['milvus']
        self.logger = logging.getLogger(__name__)
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.config['host'],
            port=self.config['port']
        )
        
        self.collection_name = self.config['collection_name']
        self.collection = None
        
        # Setup collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup or load the Light Token collection."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.logger.info(f"Loaded existing collection: {self.collection_name}")
        else:
            self._create_collection()
    
    def _create_collection(self):
        """Create the Light Token collection with proper schema."""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="token_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="joint_embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="spectral_power", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Light Token collection for A-LMI system"
        )
        
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index for vector search
        index_params = {
            "metric_type": self.config.get('metric_type', 'L2'),
            "index_type": self.config.get('index_type', 'IVF_FLAT'),
            "params": self.config.get('index_params', {"nlist": 1024})
        }
        
        self.collection.create_index(
            field_name="joint_embedding",
            index_params=index_params
        )
        
        # Also index spectral power for frequency-based search
        self.collection.create_index(
            field_name="spectral_power",
            index_params=index_params
        )
        
        self.logger.info(f"Created new collection: {self.collection_name}")
    
    def insert_token(self, token: LightToken):
        """
        Insert a Light Token into the vector database.
        
        Args:
            token: Light Token to store
        """
        if token.joint_embedding is None:
            self.logger.warning(f"Token {token.token_id} has no embedding, skipping")
            return
        
        # Prepare data
        data = [{
            "id": token.token_id,
            "token_id": token.token_id,
            "modality": token.modality,
            "timestamp": token.timestamp,
            "content_text": token.content_text[:65535] if token.content_text else "",
            "joint_embedding": token.joint_embedding.tolist(),
            "spectral_power": token.get_spectral_power().tolist(),
            "metadata": token.metadata
        }]
        
        # Insert
        self.collection.insert(data)
        self.collection.flush()
        
        self.logger.info(f"Inserted token {token.token_id[:8]} into vector DB")
    
    def search_semantic(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_expr: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search by semantic similarity using joint embedding.
        
        Args:
            query_embedding: 1536-dimensional query vector
            limit: Number of results to return
            filter_expr: Optional filter expression
            
        Returns:
            List of search results with similarity scores
        """
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="joint_embedding",
            param=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["token_id", "modality", "timestamp", "content_text", "metadata"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'token_id': hit.entity.get('token_id'),
                    'modality': hit.entity.get('modality'),
                    'timestamp': hit.entity.get('timestamp'),
                    'content_text': hit.entity.get('content_text'),
                    'metadata': hit.entity.get('metadata'),
                    'distance': float(hit.distance),
                    'score': 1.0 / (1.0 + hit.distance)  # Convert distance to similarity
                })
        
        return formatted_results
    
    def search_spectral(
        self,
        query_spectral_power: np.ndarray,
        limit: int = 10,
        filter_expr: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search by spectral similarity using frequency signatures.
        
        This enables frequency-based retrieval - finding information that
        operates at cyclible vibrational frequencies (resonance-based search).
        
        Args:
            query_spectral_power: 768-dimensional spectral power vector
            limit: Number of results to return
            filter_expr: Optional filter expression
            
        Returns:
            List of search results with similarity scores
        """
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        search_results = self.collection.search(
            data=[query_spectral_power.tolist()],
            anns_field="spectral_power",
            param=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["token_id", "modality", "timestamp", "content_text", "metadata"]
        )
        
        # Format results
        formatted_results = []
        for hits in search_results:
            for hit in hits:
                formatted_results.append({
                    'token_id': hit.entity.get('token_id'),
                    'modality': hit.entity.get('modality'),
                    'timestamp': hit.entity.get('timestamp'),
                    'content_text': hit.entity.get('content_text'),
                    'metadata': hit.entity.get('metadata'),
                    'distance': float(hit.distance),
                    'score': 1.0 / (1.0 + hit.distance)  # Convert distance to similarity
                })
        
        return formatted_results
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_spectral_power: np.ndarray,
        semantic_weight: float = 0.7,
        spectral_weight: float = 0.3,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and spectral similarity.
        
        This is the core innovation: finding information that is both
        semantically similar AND operating at compatible frequencies.
        
        Args:
            query_embedding: Semantic query vector
            query_spectral_power: Spectral query vector
            semantic_weight: Weight for semantic similarity
            spectral_weight: Weight for spectral similarity
            limit: Number of results to return
            
        Returns:
            List of search results with combined scores
        """
        # Get top candidates from each search
        semantic_results = self.search_semantic(query_embedding, limit=limit * 2)
        spectral_results = self.search_spectral(query_spectral_power, limit=limit * 2)
        
        # Combine results
        combined_scores = {}
        for result in semantic_results:
            token_id = result['token_id']
            if token_id not in combined_scores:
                combined_scores[token_id] = {'result': result, 'semantic_score': 0, 'spectral_score': 0}
            combined_scores[token_id]['semantic_score'] = result['score']
        
        for result in spectral_results:
            token_id = result['token_id']
            if token_id not in combined_scores:
                combined_scores[token_id] = {'result': result, 'semantic_score': 0, 'spectral_score': 0}
            combined_scores[token_id]['spectral_score'] = result['score']
            combined_scores[token_id]['result'] = result
        
        # Calculate weighted scores
        final_results = []
        for token_id, data in combined_scores.items():
            combined_score = (
                semantic_weight * data['semantic_score'] +
                spectral_weight * data['spectral_score']
            )
            final_results.append({
                **data['result'],
                'combined_score': combined_score
            })
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results[:limit]

