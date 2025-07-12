#!/usr/bin/env python3
"""
Semantic Search Service for Better Memory MCP
Uses ModernColBERT from Hugging Face for high-quality semantic search
"""

import json
import logging
import sys
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
import faiss
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    entity_name: str
    entity_type: str
    observation: str
    score: float
    timestamp: Optional[int] = None
    confidence: Optional[float] = None

class SemanticSearchEngine:
    def __init__(self, model_name: str = "lightonai/Reason-ModernColBERT", cache_dir: str = "./cache"):
        self.model_name = model_name
        
        # Handle cache directory with multiple fallbacks
        if not Path(cache_dir).is_absolute():
            # Try environment variable first
            if os.environ.get('BETTER_MEMORY_CACHE'):
                cache_dir = os.environ['BETTER_MEMORY_CACHE']
            else:
                # Try multiple locations
                possible_dirs = [
                    Path.home() / '.cache' / 'better-memory-mcp',  # User cache
                    Path(__file__).parent / cache_dir,  # Script directory
                    Path.cwd() / cache_dir,  # Working directory
                ]
                
                for d in possible_dirs:
                    try:
                        d.mkdir(parents=True, exist_ok=True)
                        cache_dir = str(d)
                        break
                    except:
                        continue
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Index components
        self.index = None
        self.metadata = []
        self.embeddings_cache = {}
        
        # Index persistence paths
        self.index_path = self.cache_dir / "faiss_index.bin"
        self.metadata_path = self.cache_dir / "metadata.pkl"
        self.embeddings_path = self.cache_dir / "embeddings.pkl"
        
        logger.info(f"Initialized semantic search engine with device: {self.device}")
    
    def load_model(self):
        """Load the ModernColBERT model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts using ModernColBERT"""
        if not self.model:
            self.load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # For ModernColBERT, we need to handle the specific output format
                # The model returns contextualized embeddings
                if hasattr(outputs, 'last_hidden_state'):
                    # Use mean pooling over token embeddings
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # Expand attention mask for broadcasting
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    
                    # Apply attention mask
                    embeddings = embeddings * input_mask_expanded
                    
                    # Sum and normalize
                    sum_embeddings = torch.sum(embeddings, 1)
                    sum_mask = input_mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    # Fallback for different model architectures
                    embeddings = outputs[0]
                
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def build_index(self, entities: List[Dict], force_rebuild: bool = False):
        """Build or update the FAISS index from entities"""
        if not force_rebuild and self.index_path.exists():
            logger.info("Loading existing index...")
            self.load_index()
            return
        
        logger.info("Building semantic search index...")
        
        # Extract all observations with metadata
        texts = []
        metadata = []
        
        for entity in entities:
            entity_name = entity['name']
            entity_type = entity['entityType']
            observations = entity.get('observations', [])
            
            for obs in observations:
                # Handle both string and object observation formats
                if isinstance(obs, str):
                    text = obs
                    timestamp = None
                    confidence = None
                else:
                    text = obs.get('content', '')
                    timestamp = obs.get('timestamp')
                    confidence = obs.get('confidence')
                
                # Create searchable text combining entity info and observation
                searchable_text = f"{entity_name} ({entity_type}): {text}"
                texts.append(searchable_text)
                
                metadata.append({
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'observation': text,
                    'timestamp': timestamp,
                    'confidence': confidence
                })
        
        if not texts:
            logger.warning("No observations to index")
            return
        
        # Encode all texts
        logger.info(f"Encoding {len(texts)} observations...")
        embeddings = self.encode_text(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        self.metadata = metadata
        
        # Save index and metadata
        self.save_index()
        
        logger.info(f"Index built successfully with {len(texts)} observations")
    
    def incremental_update(self, new_entities: List[Dict]):
        """Add new entities to existing index"""
        if not self.index:
            # No existing index, build from scratch
            self.build_index(new_entities)
            return
        
        logger.info("Performing incremental index update...")
        
        texts = []
        new_metadata = []
        
        # Extract observations from new entities
        for entity in new_entities:
            entity_name = entity['name']
            entity_type = entity['entityType']
            observations = entity.get('observations', [])
            
            for obs in observations:
                if isinstance(obs, str):
                    text = obs
                    timestamp = None
                    confidence = None
                else:
                    text = obs.get('content', '')
                    timestamp = obs.get('timestamp')
                    confidence = obs.get('confidence')
                
                searchable_text = f"{entity_name} ({entity_type}): {text}"
                texts.append(searchable_text)
                
                new_metadata.append({
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'observation': text,
                    'timestamp': timestamp,
                    'confidence': confidence
                })
        
        if texts:
            # Encode new texts
            embeddings = self.encode_text(texts)
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            self.metadata.extend(new_metadata)
            
            # Save updated index
            self.save_index()
            
            logger.info(f"Added {len(texts)} new observations to index")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar observations"""
        if not self.index:
            logger.error("No index available. Please build index first.")
            return []
        
        # Encode query
        query_embedding = self.encode_text([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(SearchResult(
                    entity_name=meta['entity_name'],
                    entity_type=meta['entity_type'],
                    observation=meta['observation'],
                    score=float(score),
                    timestamp=meta.get('timestamp'),
                    confidence=meta.get('confidence')
                ))
        
        return results
    
    def save_index(self):
        """Save index and metadata to disk"""
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {self.metadata_path}")
    
    def load_index(self):
        """Load index and metadata from disk"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index from {self.index_path}")
        
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded metadata from {self.metadata_path}")

class SemanticSearchService:
    """Service to handle requests from Node.js"""
    
    def __init__(self, memory_file_path: str = "memory.json"):
        # Handle relative and absolute paths
        if not Path(memory_file_path).is_absolute():
            # Try multiple locations for relative paths
            possible_paths = [
                Path(memory_file_path),  # Current directory
                Path(__file__).parent / memory_file_path,  # Script directory
                Path.cwd() / memory_file_path,  # Working directory
            ]
            
            for p in possible_paths:
                if p.exists():
                    memory_file_path = str(p.absolute())
                    logger.info(f"Found memory file at: {memory_file_path}")
                    break
            else:
                # Use first path as default even if it doesn't exist yet
                memory_file_path = str(possible_paths[0].absolute())
                logger.warning(f"Memory file not found, will use: {memory_file_path}")
        
        self.memory_file_path = memory_file_path
        self.engine = SemanticSearchEngine()
        self.last_modified = None
    
    def load_memory_graph(self) -> List[Dict]:
        """Load entities from memory.json"""
        entities = []
        
        try:
            with open(self.memory_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if item.get('type') == 'entity' and not item.get('archived') and not item.get('merged'):
                            entities.append(item)
        except FileNotFoundError:
            logger.warning(f"Memory file not found: {self.memory_file_path}")
        except Exception as e:
            logger.error(f"Error loading memory file: {e}")
        
        return entities
    
    def check_and_update_index(self):
        """Check if memory file has been modified and update index if needed"""
        try:
            current_modified = Path(self.memory_file_path).stat().st_mtime
            
            if self.last_modified is None or current_modified > self.last_modified:
                logger.info("Memory file has been modified, updating index...")
                entities = self.load_memory_graph()
                self.engine.build_index(entities, force_rebuild=True)
                self.last_modified = current_modified
        except Exception as e:
            logger.error(f"Error checking memory file: {e}")
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle search request from Node.js"""
        try:
            action = request.get('action')
            
            if action == 'search':
                # Check for index updates
                self.check_and_update_index()
                
                query = request.get('query', '')
                k = request.get('k', 10)
                threshold = request.get('threshold', 0.0)
                
                results = self.engine.search(query, k, threshold)
                
                return {
                    'success': True,
                    'results': [
                        {
                            'entity_name': r.entity_name,
                            'entity_type': r.entity_type,
                            'observation': r.observation,
                            'score': r.score,
                            'timestamp': r.timestamp,
                            'confidence': r.confidence
                        }
                        for r in results
                    ]
                }
            
            elif action == 'rebuild_index':
                entities = self.load_memory_graph()
                self.engine.build_index(entities, force_rebuild=True)
                return {'success': True, 'message': f'Index rebuilt with {len(entities)} entities'}
            
            elif action == 'ping':
                return {'success': True, 'message': 'Semantic search service is running'}
            
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
        
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {'success': False, 'error': str(e)}

async def main():
    """Main entry point for the semantic search service"""
    # Get memory file path from environment or use default
    import os
    memory_file_path = os.environ.get('MEMORY_FILE_PATH', 'memory.json')
    
    # Log startup information for debugging
    logger.info(f"Starting semantic search service...")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    logger.info(f"Memory file path: {memory_file_path}")
    
    service = SemanticSearchService(memory_file_path)
    
    logger.info("Semantic search service starting...")
    logger.info("Loading model (this may take a moment on first run)...")
    
    # Pre-load the model
    service.engine.load_model()
    
    # Initial index build
    logger.info("Building initial index...")
    service.check_and_update_index()
    
    logger.info("Service ready, listening for requests on stdin...")
    
    # Read requests from stdin (JSONL format)
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            
            request = json.loads(line.decode().strip())
            response = await service.handle_request(request)
            
            # Add request_id to response if it was in the request
            if 'request_id' in request:
                response['request_id'] = request['request_id']
            
            # Write response to stdout
            print(json.dumps(response), flush=True)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            print(json.dumps({'success': False, 'error': 'Invalid JSON'}), flush=True)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(json.dumps({'success': False, 'error': str(e)}), flush=True)

if __name__ == '__main__':
    asyncio.run(main())