# MCP Integration Guide for Semantic Search

## The Path Problem

When Claude Desktop runs the MCP server, it executes from a different working directory than where your files are located. This causes issues with:
- Finding the Python script
- Locating the memory.json file
- Accessing the model cache directory

## Solution 1: Using NPX (Recommended for Development)

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
        "PYTHONPATH": "/path/to/better-memory-mcp",
        "PATH": "/usr/local/bin:/usr/bin:/bin:/path/to/your/python/bin"
      }
    }
  }
}
```

## Solution 2: Local Installation with Absolute Paths

First, install globally or in a known location:
```bash
cd /Users/yourusername/better-memory-mcp
npm install -g .
```

Then in claude_desktop_config.json:
```json
{
  "mcpServers": {
    "memory": {
      "command": "better-memory-mcp",
      "args": [],
      "env": {
        "MEMORY_FILE_PATH": "/Users/yourusername/.claude/memory.json",
        "BETTER_MEMORY_CACHE": "/Users/yourusername/.claude/memory-cache",
        "PYTHONPATH": "/Users/yourusername/better-memory-mcp",
        "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"
      }
    }
  }
}
```

## Solution 3: Wrapper Script (Most Reliable)

Create a wrapper script at `/Users/yourusername/bin/better-memory-wrapper.sh`:

```bash
#!/bin/bash
# Better Memory MCP Wrapper Script

# Set the installation directory
INSTALL_DIR="/Users/yourusername/better-memory-mcp"

# Set memory file location
export MEMORY_FILE_PATH="${MEMORY_FILE_PATH:-$HOME/.claude/memory.json}"

# Ensure Python environment is available
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:$PATH"

# Activate Python virtual environment if it exists
if [ -f "$INSTALL_DIR/venv/bin/activate" ]; then
    source "$INSTALL_DIR/venv/bin/activate"
fi

# Set Python path for imports
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"

# Change to installation directory for relative imports
cd "$INSTALL_DIR"

# Run the actual MCP server
exec node "$INSTALL_DIR/dist/index.js"
```

Make it executable:
```bash
chmod +x /Users/yourusername/bin/better-memory-wrapper.sh
```

Then in claude_desktop_config.json:
```json
{
  "mcpServers": {
    "memory": {
      "command": "/Users/yourusername/bin/better-memory-wrapper.sh",
      "args": []
    }
  }
}
```

## Solution 4: Docker (Production Ready)

Update the Dockerfile to include Python:
```dockerfile
FROM node:22-alpine AS builder
# ... existing build steps ...

FROM python:3.11-slim-bookworm AS release

# Install Node.js
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Node app
COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/package*.json /app/
COPY semantic_search.py requirements.txt /app/

WORKDIR /app

# Install Node dependencies
RUN npm ci --only=production

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('lightonai/Reason-ModernColBERT', cache_dir='/app/cache'); \
    AutoModel.from_pretrained('lightonai/Reason-ModernColBERT', cache_dir='/app/cache', trust_remote_code=True)"

ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["node", "dist/index.js"]
```

## Debugging Tips

1. **Check Python availability**:
   Add this to your semantic_search_bridge.ts constructor:
   ```typescript
   console.error('Python check:', process.env.PATH);
   console.error('Working directory:', process.cwd());
   ```

2. **Test memory file access**:
   ```bash
   # In your MCP config, add:
   "env": {
     "DEBUG": "true",
     "MEMORY_FILE_PATH": "/absolute/path/to/memory.json"
   }
   ```

3. **Common issues**:
   - Python not in PATH: Add Python location explicitly
   - Virtual env not activated: Use wrapper script
   - Model cache permissions: Ensure write access to cache dir
   - Memory file not found: Use absolute paths

## Platform-Specific Notes

### macOS
- Python might be at `/opt/homebrew/bin/python3` (Apple Silicon) or `/usr/local/bin/python3` (Intel)
- Add both to PATH to be safe

### Windows
- Use forward slashes in paths even on Windows
- Python might be at `C:/Users/USERNAME/AppData/Local/Programs/Python/Python311`
- Consider using WSL2 for better compatibility

### Linux
- Usually works out of the box if Python is in system PATH
- May need to install python3-venv package

## Quick Test

To verify your setup works:
```bash
# Test if Python can be found and imports work
cd /path/to/better-memory-mcp
python3 semantic_search.py
# Should see "Semantic search service starting..."
```

If this works but Claude Desktop doesn't, it's a PATH/environment issue.