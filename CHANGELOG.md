# Changelog

## [0.8.0] - 2025-01-12

### Added
- **Semantic Search**: Integrated ModernColBERT neural embeddings for advanced semantic search capabilities
  - New `semantic_search` tool that understands meaning and context, not just keywords
  - Automatic indexing of all existing memories on startup
  - Real-time index updates as new memories are added
  - Graceful fallback to keyword search if Python dependencies unavailable
- **Python-Node.js Bridge**: Robust communication between TypeScript MCP server and Python ML service
- **Setup Script**: Automated setup for Python environment and model downloads
- **Test Suite**: Comprehensive testing for semantic search functionality

### Enhanced
- Search results now include similarity scores and can be filtered by threshold
- Documentation updated with semantic search examples and setup instructions
- Error handling improved with informative fallback behavior

### Technical
- Added ModernColBERT model from Hugging Face (lightonai/Reason-ModernColBERT)
- FAISS vector database for efficient similarity search
- Asynchronous Python service with JSONL communication protocol
- Pre-download of model weights for faster first-time startup

## [0.7.2] - Previous Release

### Features
- Temporal tracking with timestamps
- Confidence scores for observations
- Entity archiving and merging
- Recent changes view
- Backward compatibility with legacy data formats