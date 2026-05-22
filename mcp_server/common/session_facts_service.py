"""
Session Facts Service - Python MongoDB Client

This module provides access to the session facts stored in MongoDB.
It mirrors the functionality of the Node.js sessionFactsService and sessionMemoryService.
"""

import json
import os
from typing import Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# MongoDB Configuration - Load from this codebase's config.json
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'config',
    'config.json'
)

COLLECTION_NAME = "session_memory"


class SessionFactsService:
    """Service for retrieving session facts from MongoDB."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the session facts service.
        
        Args:
            config_path: Optional path to mongodb_config.json
        """
        self._client = None
        self._db = None
        self._config = self._load_config(config_path or CONFIG_PATH)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MongoDB configuration from config.json file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Extract just the mongodb section
                return config.get('mongodb', {})
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            raise
        except KeyError:
            print(f"Error: 'mongodb' section not found in config file")
            raise
    
    def connect(self):
        """Establish connection to MongoDB."""
        if self._client is not None:
            return  # Already connected
        
        try:
            mongo_url = self._config.get("url")
            connection_options = self._config.get("connection_options", {})
            
            self._client = MongoClient(mongo_url, **connection_options)
            self._db = self._client[self._config.get("database", "copilot")]
            
            # Test connection
            self._client.admin.command('ping')
            print(f"[SessionFacts] Connected to MongoDB: {self._config.get('database')}")
        except ConnectionFailure as e:
            print(f"[SessionFacts] Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("[SessionFacts] Disconnected from MongoDB")
    
    def get_session_memory(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve session memory for a given session.
        
        Args:
            session_id: The session ID to retrieve
            user_id: Optional user ID for validation
            
        Returns:
            Dictionary containing session memory with facts, focus, entities, etc.
        """
        if not session_id:
            return self._default_session_memory(None, user_id)
        
        try:
            self.connect()
            collection = self._db[COLLECTION_NAME]
            
            query = {"session_id": session_id}
            if user_id:
                query["user_id"] = user_id
            
            doc = collection.find_one(query)
            
            if not doc:
                return self._default_session_memory(session_id, user_id)
            
            # Remove MongoDB _id field
            if '_id' in doc:
                del doc['_id']
            
            return {
                **self._default_session_memory(session_id, user_id),
                **doc
            }
        except (ConnectionFailure, OperationFailure) as e:
            print(f"[SessionFacts] Error retrieving session memory: {e}")
            return self._default_session_memory(session_id, user_id)
    
    def _default_session_memory(self, session_id: Optional[str], user_id: Optional[str]) -> Dict[str, Any]:
        """Return default session memory structure."""
        return {
            "session_id": session_id,
            "user_id": user_id,
            "focus": None,
            "facts": {},
            "tool_facts": {},
            "entities": {},
            "last_tool": None,
            "updated_at": None
        }
    
    def get_session_facts(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve just the facts from session memory.
        
        Args:
            session_id: The session ID to retrieve
            user_id: Optional user ID for validation
            
        Returns:
            Dictionary containing session facts
        """
        memory = self.get_session_memory(session_id, user_id)
        return memory.get("facts", {})
    
    def format_session_facts(self, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Format session facts as a string for inclusion in LLM prompts.
        
        Args:
            session_id: The session ID to retrieve
            user_id: Optional user ID for validation
            
        Returns:
            Formatted string representation of session facts
        """
        memory = self.get_session_memory(session_id, user_id)
        facts = memory.get("facts", {})
        focus = memory.get("focus")
        entities = memory.get("entities", {})
        
        if not facts or len(facts) == 0:
            return "No session facts available."
        
        parts = []
        
        if focus:
            parts.append(f"CURRENT FOCUS:\n{json.dumps(focus, indent=2)}")
            parts.append("")
        
        parts.append(f"SESSION FACTS:\n{json.dumps(facts, indent=2)}")
        
        # Include relevant entities if available
        if entities:
            relevant_entities = {}
            for entity_id, entity_data in entities.items():
                if isinstance(entity_data, dict):
                    # Include entities that might be useful for workflow generation
                    if entity_data.get("type") in ["file", "genome", "sample"]:
                        relevant_entities[entity_id] = entity_data
            
            if relevant_entities:
                parts.append("")
                parts.append(f"SESSION ENTITIES:\n{json.dumps(relevant_entities, indent=2)}")
        
        return "\n\n".join(parts)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Global instance for convenience
_global_service = None


def get_session_facts_service() -> SessionFactsService:
    """Get or create the global session facts service instance."""
    global _global_service
    if _global_service is None:
        _global_service = SessionFactsService()
    return _global_service


def get_session_facts(session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get session facts.
    
    Args:
        session_id: The session ID to retrieve
        user_id: Optional user ID for validation
        
    Returns:
        Dictionary containing session facts
    """
    service = get_session_facts_service()
    return service.get_session_facts(session_id, user_id)


def format_session_facts_for_llm(session_id: str, user_id: Optional[str] = None) -> str:
    """
    Convenience function to format session facts for LLM prompts.
    
    Args:
        session_id: The session ID to retrieve
        user_id: Optional user ID for validation
        
    Returns:
        Formatted string representation of session facts
    """
    return ""  # Session facts disabled
    service = get_session_facts_service()
    return service.format_session_facts(session_id, user_id)

