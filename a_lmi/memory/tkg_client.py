"""
Temporal Knowledge Graph Client for Neo4j

Stores entities and relationships with temporal information for time-aware reasoning.
"""

from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.light_token import LightToken


class TKGClient:
    """
    Client for Neo4j temporal knowledge graph operations.
    
    Extracts entities and relationships from Light Tokens and stores them
    with temporal annotations for time-aware reasoning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Neo4j driver.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['infrastructure']['neo4j']
        self.logger = logging.getLogger(__name__)
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.config['username'], self.config['password'])
        )
        
        # Verify connection
        with self.driver.session() as session:
            session.run("RETURN 1")
        
        self.logger.info(f"Connected to Neo4j at {self.uri}")
    
    @property
    def uri(self) -> str:
        """Get Neo4j URI."""
        return self.config['uri']
    
    def store_entities_from_token(self, token: LightToken):
        """
        Extract and store entities from a Light Token.
        
        In production, this would use NER (Named Entity Recognition) to
        extract entities. For now, we create a simple relationship to the token.
        
        Args:
            token: Light Token containing entities
        """
        if not token.content_text:
            self.logger.debug(f"Token {token.token_id} has no content_text")
            return
        
        timestamp = token.timestamp
        
        # Create a node for this Light Token
        query = """
        MERGE (t:LightToken {token_id: $token_id})
        SET t.modality = $modality,
            t.timestamp = $timestamp,
            t.content_text = $content_text,
            t.source_uri = $source_uri
        RETURN t
        """
        
        with self.driver.session() as session:
            session.run(
                query,
                token_id=token.token_id,
                modality=token.modality,
                timestamp=timestamp,
                content_text=token.content_text[:1000],  # Limit size
                source_uri=token.source_uri
            )
        
        # Extract entities using NER service
        from ..services.ner_service import NERService
        ner = NERService()
        entities = ner.extract_entities(token.content_text)
        
        # Store entities and relationships
        for entity in entities:
            # Create entity node
            entity_query = f"""
            MERGE (e:{entity['label']} {{name: $name}})
            SET e.description = $description
            RETURN e
            """
            
            session.run(
                entity_query,
                name=entity['text'],
                description=entity.get('description', entity['label'])
            )
            
            # Create relationship from token to entity
            rel_query = f"""
            MATCH (t:LightToken {{token_id: $token_id}})
            MATCH (e:{entity['label']} {{name: $entity_name}})
            MERGE (t)-[r:CONTAINS_ENTITY]->(e)
            SET r.timestamp = $timestamp
            RETURN r
            """
            
            session.run(
                rel_query,
                token_id=token.token_id,
                entity_name=entity['text'],
                timestamp=timestamp
            )
        
        self.logger.info(f"Stored {len(entities)} entities from token {token.token_id[:8]} in TKG")
    
    def create_entity(
        self,
        entity_type: str,
        entity_name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Create an entity node in the knowledge graph.
        
        Args:
            entity_type: Type of entity (Person, Organization, Concept, etc.)
            entity_name: Name of the entity
            properties: Additional properties
        """
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        SET e += $properties
        RETURN e
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                name=entity_name,
                properties=properties or {}
            )
            return result.single()
    
    def create_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Create a relationship between two entities.
        
        Args:
            from_entity: Name of source entity
            to_entity: Name of target entity
            relationship_type: Type of relationship
            properties: Additional properties (including timestamp)
        """
        query = f"""
        MATCH (a {{name: $from_name}}), (b {{name: $to_name}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        RETURN r
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                from_name=from_entity,
                to_name=to_entity,
                properties=properties or {}
            )
            return result.single()
    
    def query_entities(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities by type.
        
        Args:
            entity_type: Type of entity to query
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        query = f"MATCH (e:{entity_type}) RETURN e LIMIT $limit"
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record['e']) for record in result]
    
    def temporal_query(
        self,
        entity_name: str,
        relationship_type: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query relationships with temporal constraints.
        
        Args:
            entity_name: Name of entity
            relationship_type: Type of relationship
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            
        Returns:
            List of relationships matching temporal criteria
        """
        query = f"""
        MATCH (a {{name: $entity_name}})-[r:{relationship_type}]->(b)
        WHERE ($start_time IS NULL OR r.timestamp >= $start_time)
          AND ($end_time IS NULL OR r.timestamp <= $end_time)
        RETURN r, b
        ORDER BY r.timestamp DESC
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                entity_name=entity_name,
                relationship_type=relationship_type,
                start_time=start_time,
                end_time=end_time
            )
            return [{
                'relationship': dict(record['r']),
                'target': dict(record['b'])
            } for record in result]
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        self.logger.info("Closed Neo4j connection")

