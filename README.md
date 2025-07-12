# Better Memory MCP Server

An enhanced knowledge graph memory system for Claude with temporal tracking, confidence scores, entity archiving, semantic search, and advanced management capabilities. This server enables Claude to maintain persistent memory across conversations with improved organization and data quality features.

**Original implementation by [Anthropic, PBC](https://anthropic.com)**  
**Enhanced by [@sockeye44](https://github.com/sockeye44) and Claude Opus**

## Enhanced Features

This enhanced version includes several improvements over the original memory server:

- **Semantic Search**: Advanced neural search using ModernColBERT embeddings for understanding meaning and context
- **Temporal Tracking**: All observations and entities now include timestamps for when they were created
- **Confidence Scores**: Observations can have confidence scores (0-1) to indicate certainty levels
- **Entity Archiving**: Soft-delete functionality allows entities to be archived rather than permanently deleted
- **Entity Merging**: Consolidate duplicate entities while preserving all observations and relations
- **Recent Changes View**: Query recent activity within a specified time window
- **Automatic Backfilling**: Existing memories are automatically indexed for semantic search on first use
- **Backward Compatibility**: Seamlessly handles legacy data format while using enhanced features for new data

## Core Concepts

### Entities
Entities are the primary nodes in the knowledge graph. Each entity has:
- A unique name (identifier)
- An entity type (e.g., "person", "organization", "event")
- A list of observations

Example:
```json
{
  "name": "John_Smith",
  "entityType": "person",
  "observations": ["Speaks fluent Spanish"]
}
```

### Relations
Relations define directed connections between entities. They are always stored in active voice and describe how entities interact or relate to each other.

Example:
```json
{
  "from": "John_Smith",
  "to": "Anthropic",
  "relationType": "works_at"
}
```
### Observations
Observations are discrete pieces of information about an entity with enhanced metadata:

- Stored with timestamps and confidence scores
- Attached to specific entities
- Can be added or removed independently
- Should be atomic (one fact per observation)

Example:
```json
{
  "entityName": "John_Smith",
  "observations": [
    {
      "content": "Speaks fluent Spanish",
      "timestamp": 1734567890123,
      "confidence": 0.95
    },
    {
      "content": "Graduated in 2019",
      "timestamp": 1734567890124,
      "confidence": 1.0
    }
  ]
}
```

## API

### Tools
- **create_entities**
  - Create multiple new entities in the knowledge graph
  - Input: `entities` (array of objects)
    - Each object contains:
      - `name` (string): Entity identifier
      - `entityType` (string): Type classification
      - `observations` (string[]): Associated observations
  - Ignores entities with existing names

- **create_relations**
  - Create multiple new relations between entities
  - Input: `relations` (array of objects)
    - Each object contains:
      - `from` (string): Source entity name
      - `to` (string): Target entity name
      - `relationType` (string): Relationship type in active voice
  - Skips duplicate relations

- **add_observations**
  - Add new observations to existing entities with optional confidence scores
  - Input: `observations` (array of objects)
    - Each object contains:
      - `entityName` (string): Target entity
      - `contents` (string[]): New observations to add
      - `confidence` (number[], optional): Confidence scores (0-1) for each observation
  - Returns added observations per entity
  - Fails if entity doesn't exist

- **delete_entities**
  - Remove entities and their relations
  - Input: `entityNames` (string[])
  - Cascading deletion of associated relations
  - Silent operation if entity doesn't exist

- **delete_observations**
  - Remove specific observations from entities
  - Input: `deletions` (array of objects)
    - Each object contains:
      - `entityName` (string): Target entity
      - `observations` (string[]): Observations to remove
  - Silent operation if observation doesn't exist

- **delete_relations**
  - Remove specific relations from the graph
  - Input: `relations` (array of objects)
    - Each object contains:
      - `from` (string): Source entity name
      - `to` (string): Target entity name
      - `relationType` (string): Relationship type
  - Silent operation if relation doesn't exist

- **read_graph**
  - Read the knowledge graph with controllable detail
  - Input:
    - `detailLevel` (string, optional): "minimal", "summary", or "full" (default: "summary")
    - `entityNames` (string[], optional): Get full details for specific entities only
    - `includeArchived` (boolean, optional): Include archived entities (default: false)
  - Returns graph structure based on detail level

- **search_nodes**
  - Search for nodes based on keyword matching
  - Input: `query` (string)
  - Searches across:
    - Entity names
    - Entity types
    - Observation content
  - Returns matching entities and their relations

- **semantic_search** *(NEW)*
  - Advanced semantic search using neural embeddings
  - Input:
    - `query` (string): Natural language search query
    - `k` (number, optional): Number of results (default 10)
    - `threshold` (number, optional): Minimum similarity score 0-1 (default 0)
  - Uses ModernColBERT model to understand meaning and context
  - Returns entities ranked by semantic similarity
  - Automatically falls back to keyword search if unavailable

- **open_nodes**
  - Retrieve specific nodes by name
  - Input: `names` (string[])
  - Returns:
    - Requested entities
    - Relations between requested entities
  - Silently skips non-existent nodes

- **merge_entities**
  - Merge one entity into another, combining observations and updating relations
  - Input:
    - `sourceName` (string): Entity to merge from (will be deleted)
    - `targetName` (string): Entity to merge into (will be preserved)
  - Combines all unique observations and redirects all relations
  - Returns success status and merge summary

- **archive_entity**
  - Archive an entity (soft delete - hidden from normal queries)
  - Input: `entityName` (string)
  - Archived entities are excluded from standard read operations
  - Can be unarchived later

- **unarchive_entity**
  - Restore a previously archived entity
  - Input: `entityName` (string)
  - Makes the entity visible in normal queries again

- **get_recent_changes**
  - Get entities, relations, and observations created/modified within specified time
  - Input: `hours` (number, optional): Hours to look back (default: 24)
  - Returns:
    - Recently created entities
    - Recently created relations
    - Entities with recent observations

## Setup Instructions

### Quick Setup (Recommended)

For the full experience including semantic search:

```bash
# Clone the repository
git clone https://github.com/sockeye44/better-memory-mcp
cd better-memory-mcp

# Run the setup script
./setup.sh
```

The setup script will:
- Check Python 3.8+ is installed
- Create a Python virtual environment
- Install all Python dependencies including PyTorch and ModernColBERT
- Pre-download the neural model for faster first use
- Install Node.js dependencies
- Build the TypeScript code

### Manual Setup

If you prefer to set up manually or the script fails:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Build TypeScript
npm run build
```

**Note**: Semantic search requires Python 3.8+ and PyTorch. If these are not available, the server will still work with keyword search only.

# Usage with Claude Desktop

### Setup

**Important**: For semantic search to work in Claude Desktop, you need to ensure Python is accessible. See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) for detailed setup instructions.

#### Quick Setup for Claude Desktop

1. **Option A: Using absolute paths** (most reliable):
```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/absolute/path/to/better-memory-mcp/dist/index.js"],
      "env": {
        "MEMORY_FILE_PATH": "/Users/yourusername/.claude/memory.json",
        "BETTER_MEMORY_DIR": "/absolute/path/to/better-memory-mcp",
        "PATH": "/absolute/path/to/better-memory-mcp/venv/bin:$PATH"
      }
    }
  }
}
```

2. **Option B: Standard config** (if Python is in system PATH):

Add this to your claude_desktop_config.json:

#### Docker

```json
{
  "mcpServers": {
    "memory": {
      "command": "docker",
      "args": ["run", "-i", "-v", "claude-memory:/app/dist", "--rm", "mcp/better-memory"]
    }
  }
}
```

#### NPX
```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@sockeye44/better-memory-mcp"
      ]
    }
  }
}
```

#### NPX with custom setting

The server can be configured using the following environment variables:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@sockeye44/better-memory-mcp"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/path/to/custom/memory.json"
      }
    }
  }
}
```

- `MEMORY_FILE_PATH`: Path to the memory storage JSON file (default: `memory.json` in the server directory)

# VS Code Installation Instructions

For quick installation, use one of the one-click installation buttons below:

[![Install with NPX in VS Code](https://img.shields.io/badge/VS_Code-NPM-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=memory&config=%7B%22command%22%3A%22npx%22%2C%22args%22%3A%5B%22-y%22%2C%22%40sockeye44%2Fbetter-memory-mcp%22%5D%7D) [![Install with NPX in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-NPM-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=memory&config=%7B%22command%22%3A%22npx%22%2C%22args%22%3A%5B%22-y%22%2C%22%40sockeye44%2Fbetter-memory-mcp%22%5D%7D&quality=insiders)

[![Install with Docker in VS Code](https://img.shields.io/badge/VS_Code-Docker-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=memory&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-i%22%2C%22-v%22%2C%22claude-memory%3A%2Fapp%2Fdist%22%2C%22--rm%22%2C%22mcp%2Fbetter-memory%22%5D%7D) [![Install with Docker in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Docker-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=memory&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-i%22%2C%22-v%22%2C%22claude-memory%3A%2Fapp%2Fdist%22%2C%22--rm%22%2C%22mcp%2Fbetter-memory%22%5D%7D&quality=insiders)

For manual installation, add the following JSON block to your User Settings (JSON) file in VS Code. You can do this by pressing `Ctrl + Shift + P` and typing `Preferences: Open Settings (JSON)`.

Optionally, you can add it to a file called `.vscode/mcp.json` in your workspace. This will allow you to share the configuration with others. 

> Note that the `mcp` key is not needed in the `.vscode/mcp.json` file.

#### NPX

```json
{
  "mcp": {
    "servers": {
      "memory": {
        "command": "npx",
        "args": [
          "-y",
          "@sockeye44/better-memory-mcp"
        ]
      }
    }
  }
}
```

#### Docker

```json
{
  "mcp": {
    "servers": {
      "memory": {
        "command": "docker",
        "args": [
          "run",
          "-i",
          "-v",
          "claude-memory:/app/dist",
          "--rm",
          "mcp/better-memory"
        ]
      }
    }
  }
}
```

### System Prompt

The prompt for utilizing memory depends on the use case. Changing the prompt will help the model determine the frequency and types of memories created.

Here is an example prompt for chat personalization. You could use this prompt in the "Custom Instructions" field of a [Claude.ai Project](https://www.anthropic.com/news/projects). 

```
Follow these steps for each interaction:

1. User Identification:
   - You should assume that you are interacting with default_user
   - If you have not identified default_user, proactively try to do so.

2. Memory Retrieval:
   - Always begin your chat by saying only "Remembering..." and retrieve all relevant information from your knowledge graph
   - Always refer to your knowledge graph as your "memory"

3. Memory
   - While conversing with the user, be attentive to any new information that falls into these categories:
     a) Basic Identity (age, gender, location, job title, education level, etc.)
     b) Behaviors (interests, habits, etc.)
     c) Preferences (communication style, preferred language, etc.)
     d) Goals (goals, targets, aspirations, etc.)
     e) Relationships (personal and professional relationships up to 3 degrees of separation)

4. Memory Update:
   - If any new information was gathered during the interaction, update your memory as follows:
     a) Create entities for recurring organizations, people, and significant events
     b) Connect them to the current entities using relations
     c) Store facts about them as observations
```

## Semantic Search

The semantic search feature uses the state-of-the-art ModernColBERT model from Hugging Face to provide intelligent, context-aware search capabilities:

### How It Works

1. **Automatic Indexing**: When the server starts, it automatically builds a semantic index of all your memories
2. **Neural Understanding**: Search queries are understood based on meaning, not just keywords
3. **Smart Ranking**: Results are ranked by semantic similarity, bringing the most relevant memories to the top
4. **Continuous Learning**: New memories are automatically added to the semantic index

### Example Queries

Semantic search understands context and meaning:

- "recent work on video analysis" - Finds memories about video-related projects
- "challenges with team collaboration" - Finds memories about teamwork issues
- "machine learning optimizations" - Finds memories about ML performance improvements
- "that project with the owl mascot" - Finds memories even with vague descriptions

### Performance

- First-time model download: ~500MB (cached for future use)
- Index building: ~1-2 seconds per 1000 observations
- Search latency: <100ms for most queries
- Memory usage: ~1GB with model loaded

### Fallback Behavior

If Python or the model are unavailable:
- The server continues to work normally
- Search automatically falls back to keyword matching
- All other features remain fully functional

## Building

Docker:

```sh
docker build -t mcp/better-memory -f src/memory/Dockerfile . 
```

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.
