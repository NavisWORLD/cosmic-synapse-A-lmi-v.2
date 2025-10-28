"""
Context Manager for Long-Term Conversation Memory

Tracks user preferences, past conversations, and context across sessions.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import json


class ContextManager:
    """
    Manages long-term context and user preferences.
    
    Features:
    - User preferences storage
    - Previous conversation recall
    - Contextual awareness
    - Personalization
    """
    
    def __init__(self, storage_path: str = "data/context/"):
        """
        Initialize context manager.
        
        Args:
            storage_path: Directory for storing context data
        """
        from pathlib import Path
        
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.user_profile = {}
        self.conversation_contexts = {}
        
        self.logger.info("Context manager initialized")
    
    def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Load user profile and preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile dictionary
        """
        profile_file = self.storage_path / f"{user_id}_profile.json"
        
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    self.user_profile[user_id] = json.load(f)
                return self.user_profile[user_id]
            except Exception as e:
                self.logger.error(f"Error loading profile: {e}")
        
        return {}
    
    def save_user_profile(self, user_id: str, profile: Dict[str, Any]):
        """
        Save user profile.
        
        Args:
            user_id: User identifier
            profile: Profile data
        """
        profile_file = self.storage_path / f"{user_id}_profile.json"
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(profile, f, indent=2)
            
            self.user_profile[user_id] = profile
            self.logger.info(f"Saved profile for user: {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving profile: {e}")
    
    def get_conversation_context(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Get context for a specific conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            Conversation context
        """
        key = f"{user_id}_{conversation_id}"
        return self.conversation_contexts.get(key, {})
    
    def set_conversation_context(self, user_id: str, conversation_id: str, context: Dict[str, Any]):
        """
        Set context for a conversation.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            context: Context data
        """
        key = f"{user_id}_{conversation_id}"
        self.conversation_contexts[key] = context

