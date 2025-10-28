"""
Neo4j Database Initialization Script

Sets up constraints, indexes, and initial schema for the temporal knowledge graph.
"""

import logging
from neo4j import GraphDatabase


def init_neo4j(uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "vibrational123"):
    """
    Initialize Neo4j database.
    
    Args:
        uri: Neo4j URI
        username: Username
        password: Password
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            # Create constraints
            logger.info("Creating constraints...")
            
            constraints = [
                "CREATE CONSTRAINT light_token_id IF NOT EXISTS FOR (n:LightToken) REQUIRE n.token_id IS UNIQUE",
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (n:PERSON) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (n:ORG) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT gpe_name IF NOT EXISTS FOR (n:GPE) REQUIRE n.name IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
            
            # Create indexes
            logger.info("Creating indexes...")
            
            indexes = [
                "CREATE INDEX light_token_timestamp IF NOT EXISTS FOR (n:LightToken) ON (n.timestamp)",
                "CREATE INDEX light_token_modality IF NOT EXISTS FOR (n:LightToken) ON (n.modality)",
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index[:50]}...")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
            
            logger.info("Neo4j initialization complete")
            
    finally:
        driver.close()


if __name__ == "__main__":
    init_neo4j()

