"""ollama_commands.py

This module contains the class OllamaCommands which provides simplified methods for 
interacting with the ollama library. It focuses on core functionality for normal LLM 
prompts and vision prompts with model selection.

Author: @BorcherdingL
Date: 6/23/2025
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

# Setup logging
logger = logging.getLogger(__name__)

class OllamaCommands:
    
    def __init__(self):
        self.name = "ollamaCommands"
        
        # Initialize ollama
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            logger.error("Ollama library not available")
            self.ollama = None
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        if not self.ollama:
            raise Exception("Ollama not available")
        
        try:
            logger.info("Getting available models")
            result = self.ollama.list()
            
            # Handle new ListResponse format
            if hasattr(result, 'models'):
                return result.models
            return result
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
    
    async def show_model_info(self, model: str) -> Dict[str, Any]:
        """Show detailed information about a model"""
        if not self.ollama:
            raise Exception("Ollama not available")
        
        try:
            return self.ollama.show(model)
        except Exception as e:
            logger.error(f"Error showing model info: {e}")
            raise
    
    async def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get list of currently loaded models"""
        if not self.ollama:
            raise Exception("Ollama not available")
        
        try:
            return self.ollama.ps()
        except Exception as e:
            logger.error(f"Error getting loaded models: {e}")
            raise
    
    async def chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Send a chat prompt to the specified model
        
        Args:
            model: Name of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response
            
        Returns:
            String response or async generator if streaming
        """
        if not self.ollama:
            raise Exception("Ollama not available")
        
        try:
            response = self.ollama.chat(
                model=model,
                messages=messages,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    async def vision_chat(
        self, 
        model: str, 
        prompt: str, 
        image_data: str,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Send a vision prompt with image to the specified model
        
        Args:
            model: Name of the vision model to use (e.g., 'llava')
            prompt: Text prompt for the image
            image_data: Base64 encoded image data
            stream: Whether to stream the response
            
        Returns:
            String response or async generator if streaming
        """
        if not self.ollama:
            raise Exception("Ollama not available")
        
        try:
            message = {
                "role": "user",
                "content": prompt,
                "images": [image_data]
            }
            
            response = self.ollama.chat(
                model=model,
                messages=[message],
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Error in vision_chat: {e}")
            raise
    
    async def generate(
        self, 
        model: str, 
        prompt: str, 
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Generate text using a simple prompt (alternative to chat)
        
        Args:
            model: Name of the model to use
            prompt: Text prompt
            stream: Whether to stream the response
            
        Returns:
            String response or async generator if streaming
        """
        if not self.ollama:
            raise Exception("Ollama not available")
        
        try:
            response = self.ollama.generate(
                model=model,
                prompt=prompt,
                stream=stream
            )
            
            if stream:
                return self._stream_generate_response(response)
            else:
                return response['response']
                
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            raise
    
    async def _stream_response(self, response):
        """Helper to handle streaming chat responses"""
        async for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    async def _stream_generate_response(self, response):
        """Helper to handle streaming generate responses"""
        async for chunk in response:
            if 'response' in chunk:
                yield chunk['response']