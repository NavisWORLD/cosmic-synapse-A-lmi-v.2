"""
Named Entity Recognition Service

Extracts entities from text and links them to knowledge graph.
Uses spaCy transformer model for high accuracy.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re


class NERService:
    """
    Named Entity Recognition using spaCy transformer model.
    
    Extracts entities and relationships from text content of Light Tokens.
    """
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        Initialize NER service.
        
        Args:
            model_name: spaCy model to use (transformer-based recommended)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.nlp = None
        self.initialized = False
        
        # Try to load spaCy model
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            self.initialized = True
            self.logger.info(f"NER service initialized with model: {model_name}")
        except OSError:
            self.logger.warning(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            self.logger.warning("NER functionality will be limited.")
        except ImportError:
            self.logger.warning("spaCy not installed. NER not available.")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with labels and positions
        """
        if not self.initialized:
            # Fallback to simple extraction
            return self._fallback_ner(text)
        
        try:
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'description': ent.label  # Human-readable description
                }
                entities.append(entity)
            
            self.logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return self._fallback_ner(text)
    
    def extract_relationships(
        self,
        text: str,
        source_entity: str,
        target_entity: str
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using dependency parsing.
        
        Args:
            text: Input text
            source_entity: Source entity text
            target_entity: Target entity text
            
        Returns:
            List of relationship dictionaries
        """
        if not self.initialized:
            return []
        
        try:
            doc = self.nlp(text)
            
            relationships = []
            
            # Find entity mentions
            for token in doc:
                # Check if token is related to source or target
                if source_entity.lower() in token.text.lower():
                    # Look for verbs connecting to target entity
                    for child in token.children:
                        if child.dep_ in ['nsubj', 'dobj', 'pobj']:
                            # Potential relationship
                            relationship = {
                                'source': source_entity,
                                'target': target_entity,
                                'verb': child.head.text if child.head.pos_ == 'VERB' else None,
                                'confidence': 0.5  # Simplified
                            }
                            relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {e}")
            return []
    
    def extract_temporal_info(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract temporal information from text.
        
        Args:
            text: Input text
            
        Returns:
            List of temporal expressions
        """
        if not self.initialized:
            return self._fallback_temporal(text)
        
        try:
            doc = self.nlp(text)
            
            temporal_expressions = []
            
            # Look for DATE entities
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME']:
                    temporal_expressions.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'value': self._parse_temporal_value(ent.text)
                    })
            
            return temporal_expressions
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal info: {e}")
            return self._fallback_temporal(text)
    
    def _parse_temporal_value(self, temporal_text: str) -> str:
        """
        Parse temporal expression to ISO format.
        
        Args:
            temporal_text: Temporal expression from NER
            
        Returns:
            ISO format timestamp or None
        """
        try:
            from dateutil import parser
            parsed = parser.parse(temporal_text)
            return parsed.isoformat()
        except:
            return None
    
    def _fallback_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback NER using simple heuristics.
        
        Args:
            text: Input text
            
        Returns:
            Basic entity extraction results
        """
        entities = []
        
        # Simple patterns
        # Capitalized words/phrases
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entities.append({
                'text': match.group(1),
                'label': 'PERSON',  # Guess
                'start_char': match.start(),
                'end_char': match.end(),
                'description': 'Person or organization'
            })
        
        return entities
    
    def _fallback_temporal(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback temporal extraction.
        
        Args:
            text: Input text
            
        Returns:
            Basic temporal expressions
        """
        temporal_expressions = []
        
        # Look for year patterns
        year_pattern = r'\b(19|20)\d{2}\b'
        matches = re.finditer(year_pattern, text)
        
        for match in matches:
            temporal_expressions.append({
                'text': match.group(0),
                'type': 'DATE',
                'value': match.group(0) + '-01-01'  # Approximate
            })
        
        return temporal_expressions

