"""
Kafka Topic Setup Script

Initializes all required Kafka topics with proper configurations.
"""

import logging
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


def setup_kafka_topics(bootstrap_servers: str = "localhost:9092"):
    """
    Create all required Kafka topics.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers address
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Define topics with configurations
    topics = [
        NewTopic(
            name="web_crawler",
            num_partitions=3,
            replication_factor=1,
            config={'retention.ms': '604800000'}  # 7 days
        ),
        NewTopic(
            name="audio_processing",
            num_partitions=2,
            replication_factor=1,
            config={'retention.ms': '86400000'}  # 1 day
        ),
        NewTopic(
            name="light_tokens",
            num_partitions=5,
            replication_factor=1,
            config={'retention.ms': '2592000000'}  # 30 days
        ),
        NewTopic(
            name="light_tokens_processed",
            num_partitions=5,
            replication_factor=1,
            config={'retention.ms': '2592000000'}  # 30 days
        ),
        NewTopic(
            name="reasoning",
            num_partitions=3,
            replication_factor=1,
            config={'retention.ms': '604800000'}  # 7 days
        ),
        NewTopic(
            name="web_crawler_queue",
            num_partitions=3,
            replication_factor=1,
            config={'retention.ms': '172800000'}  # 2 days
        ),
        NewTopic(
            name="cosmic_synapse",
            num_partitions=1,
            replication_factor=1,
            config={'retention.ms': '3600000'}  # 1 hour
        )
    ]
    
    try:
        # Create admin client
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='a_lmi_setup'
        )
        
        # Create topics
        admin_client.create_topics(new_topics=topics)
        logger.info("Created all Kafka topics successfully")
        
    except TopicAlreadyExistsError:
        logger.warning("Some topics already exist. Skipping...")
    except Exception as e:
        logger.error(f"Error creating topics: {e}")
        raise


if __name__ == "__main__":
    setup_kafka_topics()

