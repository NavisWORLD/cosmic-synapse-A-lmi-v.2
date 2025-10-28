"""
Milvus Database Initialization Script

Creates collections and indexes for the Light Token vector database.
"""

import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


def init_milvus():
    """Initialize Milvus collections and indexes."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Connect to Milvus
    connections.connect(
        alias="default",
        host="localhost",
        port=19530
    )
    
    collection_name = "light_tokens"
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists")
        collection = Collection(collection_name)
        collection.load()
        return
    
    # Define schema
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
    
    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema
    )
    
    # Create indexes
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    
    collection.create_index(
        field_name="joint_embedding",
        index_params=index_params
    )
    
    collection.create_index(
        field_name="spectral_power",
        index_params=index_params
    )
    
    # Load collection
    collection.load()
    
    logger.info(f"Initialized collection '{collection_name}' with indexes")


if __name__ == "__main__":
    init_milvus()

