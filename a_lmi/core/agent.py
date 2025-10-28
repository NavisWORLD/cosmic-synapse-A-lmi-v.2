"""
A-LMI Autonomous Agent Loop

Implements the perception-cognition-action cycle with autonomous learning capabilities.
"""

import yaml
import logging
from typing import Optional
from kafka import KafkaProducer, KafkaConsumer
import json
from datetime import datetime
import time

from .light_token import LightToken


class ALMIAgent:
    """
    Main autonomous agent implementing the A-LMI cognitive loop.
    
    Follows the classic agent architecture:
    - Perception: Receive data from sensors (web crawler, audio)
    - Cognition: Process into Light Tokens, store in memory, reason
    - Action: Generate responses, initiate autonomous learning
    """
    
    def __init__(self, config_path: str = "infrastructure/config.yaml"):
        """
        Initialize the A-LMI Agent.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kafka producers/consumers
        bootstrap_servers = self.config['infrastructure']['kafka']['bootstrap_servers']
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Perceptual data consumers
        self.perception_consumer = KafkaConsumer(
            'web_crawler',
            'audio_processing',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True
        )
        
        # Reasoning result consumer (for self-directed actions)
        self.reasoning_consumer = KafkaConsumer(
            'reasoning',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True
        )
        
        self.logger.info("A-LMI Agent initialized")
    
    def run_perception_loop(self):
        """
        Perception loop: Receive raw data from sensors.
        
        This runs in a separate thread/process to continuously receive
        perceptual data from web crawler and audio processor.
        """
        self.logger.info("Starting perception loop...")
        
        for message in self.perception_consumer:
            try:
                data = message.value
                topic = message.topic
                
                self.logger.debug(f"Received {topic} data: {data.get('type', 'unknown')}")
                
                # Process based on source
                if topic == 'web_crawler':
                    self._process_web_data(data)
                elif topic == 'audio_processing':
                    self._process_audio_data(data)
                    
            except Exception as e:
                self.logger.error(f"Error processing perception data: {e}", exc_info=True)
    
    def _process_web_data(self, data: dict):
        """
        Process web crawler data into Light Tokens.
        
        Args:
            data: Web page data from crawler
        """
        # Create Light Token
        token = LightToken(
            source_uri=data['url'],
            modality='text',
            raw_data_ref=data['html_ref'],
            content_text=data.get('extracted_text'),
            metadata={'crawled_at': data['timestamp'], 'domain': data.get('domain')}
        )
        
        # Send to processing core
        self._send_to_processing(token)
    
    def _process_audio_data(self, data: dict):
        """
        Process audio data into Light Tokens.
        
        Args:
            data: Audio data from microphone
        """
        # Create Light Token
        token = LightToken(
            source_uri=data.get('stream_id', 'microphone'),
            modality='audio' if data.get('type') == 'audio' else 'speech',
            raw_data_ref=data['audio_ref'],
            content_text=data.get('transcription'),
            metadata={
                'esc_class': data.get('esc_class'),
                'timestamp': data['timestamp'],
                'sample_rate': data.get('sample_rate')
            }
        )
        
        # Send to processing core
        self._send_to_processing(token)
    
    def _send_to_processing(self, token: LightToken):
        """
        Send Light Token to processing core for embedding generation.
        
        Args:
            token: Light Token to process
        """
        message = token.to_dict()
        self.producer.send('light_tokens', message)
        self.logger.debug(f"Sent token to processing: {token.token_id[:8]}")
    
    def run_cognition_loop(self):
        """
        Cognition loop: Process Light Tokens and store in memory.
        
        This receives processed Light Tokens (with embeddings) and stores
        them in the multi-layered memory system.
        """
        self.logger.info("Starting cognition loop...")
        
        # Setup consumer for processed tokens
        processed_consumer = KafkaConsumer(
            'light_tokens_processed',
            bootstrap_servers=self.config['infrastructure']['kafka']['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True
        )
        
        for message in processed_consumer:
            try:
                token_dict = message.value
                token = LightToken.from_dict(token_dict)
                
                self.logger.info(f"Processing token: {token.modality} from {token.source_uri}")
                
                # Store in memory layers (implemented in memory clients)
                self._store_in_memory(token)
                
                # Check if reasoning is triggered
                self._check_reasoning_triggers(token)
                
            except Exception as e:
                self.logger.error(f"Error in cognition loop: {e}", exc_info=True)
    
    def _store_in_memory(self, token: LightToken):
        """
        Store Light Token in all memory layers.
        
        Args:
            token: Light Token to store
        """
        # Import here to avoid circular dependencies
        from ..memory.vector_db_client import VectorDBClient
        from ..memory.object_storage_client import ObjectStorageClient
        from ..memory.tkg_client import TKGClient
        
        # Store in vector database for similarity search
        if token.joint_embedding is not None:
            vector_client = VectorDBClient(self.config)
            vector_client.insert_token(token)
        
        # Store raw data in object storage
        storage_client = ObjectStorageClient(self.config)
        storage_client.store_raw_data(token)
        
        # Extract entities and store in knowledge graph (if text content)
        if token.content_text:
            tkg_client = TKGClient(self.config)
            tkg_client.store_entities_from_token(token)
        
        self.logger.debug(f"Stored token {token.token_id[:8]} in all memory layers")
    
    def _check_reasoning_triggers(self, token: LightToken):
        """
        Check if newly stored information triggers reasoning.
        
        Args:
            token: Newly stored Light Token
        """
        # Check for contradictions, knowledge gaps, etc.
        # For now, just log
        self.logger.debug(f"Checking reasoning triggers for token {token.token_id[:8]}")
    
    def run_action_loop(self):
        """
        Action loop: Generate responses and autonomous actions.
        
        This processes reasoning results and takes actions:
        - Respond to user queries
        - Initiate new data gathering
        - Update learning goals
        """
        self.logger.info("Starting action loop...")
        
        for message in self.reasoning_consumer:
            try:
                reasoning_result = message.value
                
                self.logger.info(f"Received reasoning result: {reasoning_result.get('type')}")
                
                # Process reasoning result and take action
                if reasoning_result['type'] == 'hypothesis':
                    self._handle_hypothesis(reasoning_result)
                elif reasoning_result['type'] == 'response':
                    self._handle_response(reasoning_result)
                elif reasoning_result['type'] == 'learning_goal':
                    self._handle_learning_goal(reasoning_result)
                    
            except Exception as e:
                self.logger.error(f"Error in action loop: {e}", exc_info=True)
    
    def _handle_hypothesis(self, result: dict):
        """
        Handle a generated hypothesis by initiating investigation.
        
        Args:
            result: Hypothesis data from reasoning engine
        """
        self.logger.info(f"Handling hypothesis: {result.get('hypothesis_text')[:100]}")
        
        # Generate web crawling tasks
        urls = result.get('investigation_urls', [])
        for url in urls:
            crawler_task = {
                'url': url,
                'priority': 'high',
                'reason': 'hypothesis_investigation',
                'timestamp': datetime.now().isoformat()
            }
            self.producer.send('web_crawler_queue', crawler_task)
    
    def _handle_response(self, result: dict):
        """
        Handle a response to user query.
        
        Args:
            result: Response data from reasoning engine
        """
        self.logger.info(f"Response ready: {result.get('response_text')[:100]}")
        # Send to conversational interface
    
    def _handle_learning_goal(self, result: dict):
        """
        Handle a new autonomous learning goal.
        
        Args:
            result: Learning goal data from reasoning engine
        """
        self.logger.info(f"New learning goal: {result.get('goal_text')}")
        # Update learning plan
    
    def run(self):
        """
        Run all agent loops concurrently.
        
        This is the main entry point for the A-LMI agent.
        """
        self.logger.info("Starting A-LMI Agent main loop...")
        
        import threading
        
        # Start all loops in separate threads
        perception_thread = threading.Thread(target=self.run_perception_loop, daemon=True)
        cognition_thread = threading.Thread(target=self.run_cognition_loop, daemon=True)
        action_thread = threading.Thread(target=self.run_action_loop, daemon=True)
        
        perception_thread.start()
        cognition_thread.start()
        action_thread.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down A-LMI Agent...")


def main():
    """Main entry point."""
    agent = ALMIAgent()
    agent.run()


if __name__ == "__main__":
    main()

