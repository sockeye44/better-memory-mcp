/**
 * Semantic Search Bridge for Better Memory MCP
 * Interfaces with Python semantic search service using ModernColBERT
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';
import { fileURLToPath } from 'url';

interface SearchRequest {
  action: 'search' | 'rebuild_index' | 'ping';
  query?: string;
  k?: number;
  threshold?: number;
}

interface SearchResult {
  entity_name: string;
  entity_type: string;
  observation: string;
  score: number;
  timestamp?: number;
  confidence?: number;
}

interface SearchResponse {
  success: boolean;
  results?: SearchResult[];
  message?: string;
  error?: string;
}

export class SemanticSearchBridge extends EventEmitter {
  private pythonProcess: ChildProcess | null = null;
  private requestQueue: Map<string, (response: SearchResponse) => void> = new Map();
  private requestCounter = 0;
  private isReady = false;
  private startupPromise: Promise<void> | null = null;

  constructor(private memoryFilePath: string) {
    super();
  }

  async start(): Promise<void> {
    if (this.startupPromise) {
      return this.startupPromise;
    }

    this.startupPromise = this._start();
    return this.startupPromise;
  }

  private async _start(): Promise<void> {
    // Try multiple strategies to find the Python script
    const possiblePaths = [
      // 1. Same directory as this compiled JS file
      path.join(path.dirname(fileURLToPath(import.meta.url)), 'semantic_search.py'),
      // 2. Parent directory (in case we're in dist/)
      path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'semantic_search.py'),
      // 3. Current working directory
      path.join(process.cwd(), 'semantic_search.py'),
      // 4. Installation directory from env
      process.env.BETTER_MEMORY_DIR ? path.join(process.env.BETTER_MEMORY_DIR, 'semantic_search.py') : '',
    ].filter(p => p);
    
    let scriptPath = '';
    for (const p of possiblePaths) {
      try {
        await import('fs').then(fs => fs.promises.access(p));
        scriptPath = p;
        console.error(`Found semantic_search.py at: ${scriptPath}`);
        break;
      } catch {
        // Try next path
      }
    }
    
    if (!scriptPath) {
      throw new Error('Could not find semantic_search.py. Tried: ' + possiblePaths.join(', '));
    }
    
    // Try to find Python executable
    const pythonCommands = ['python3', 'python', '/usr/local/bin/python3', '/opt/homebrew/bin/python3'];
    let pythonCmd = 'python3';
    
    for (const cmd of pythonCommands) {
      try {
        const checkProcess = spawn(cmd, ['--version']);
        await new Promise<void>((resolve) => {
          checkProcess.on('close', (code) => {
            if (code === 0) {
              pythonCmd = cmd;
              console.error(`Using Python at: ${cmd}`);
              resolve();
            }
          });
          checkProcess.on('error', () => {});
          setTimeout(resolve, 1000); // Timeout after 1 second
        });
        break;
      } catch {
        // Try next command
      }
    }
    
    // Check if Python dependencies are installed
    const checkProcess = spawn(pythonCmd, ['-c', 'import torch, transformers, faiss']);
    
    try {
      await new Promise<void>((resolve, reject) => {
        checkProcess.on('close', (code) => {
          if (code !== 0) {
            reject(new Error('Python dependencies not installed. Please run: pip install -r requirements.txt'));
          } else {
            resolve();
          }
        });
      });
    } catch (error) {
      console.error('Semantic search dependencies check failed:', error);
      throw error;
    }

    // Start the Python process
    this.pythonProcess = spawn(pythonCmd, [scriptPath], {
      env: {
        ...process.env,
        MEMORY_FILE_PATH: this.memoryFilePath,
        PYTHONUNBUFFERED: '1'
      }
    });

    // Handle stdout (responses from Python)
    this.pythonProcess.stdout?.on('data', (data) => {
      const lines = data.toString().split('\n').filter((line: string) => line.trim());
      
      for (const line of lines) {
        try {
          const response = JSON.parse(line) as SearchResponse & { request_id?: string };
          
          // Check if this is a response to a request
          if (response.request_id && this.requestQueue.has(response.request_id)) {
            const callback = this.requestQueue.get(response.request_id)!;
            this.requestQueue.delete(response.request_id);
            callback(response);
          }
        } catch (error) {
          console.error('Failed to parse Python response:', line, error);
        }
      }
    });

    // Handle stderr (logging from Python)
    this.pythonProcess.stderr?.on('data', (data) => {
      const message = data.toString().trim();
      if (message.includes('Service ready')) {
        this.isReady = true;
        this.emit('ready');
      }
      console.error('[Python]', message);
    });

    // Handle process exit
    this.pythonProcess.on('exit', (code) => {
      console.error(`Python process exited with code ${code}`);
      this.isReady = false;
      this.pythonProcess = null;
      this.emit('exit', code);
    });

    // Wait for the service to be ready
    await new Promise<void>((resolve) => {
      if (this.isReady) {
        resolve();
      } else {
        this.once('ready', resolve);
      }
    });
  }

  async stop(): Promise<void> {
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
    this.isReady = false;
    this.requestQueue.clear();
  }

  private async sendRequest(request: SearchRequest): Promise<SearchResponse> {
    if (!this.pythonProcess || !this.isReady) {
      await this.start();
    }

    const requestId = `req_${++this.requestCounter}`;
    const fullRequest = { ...request, request_id: requestId };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.requestQueue.delete(requestId);
        reject(new Error('Request timeout'));
      }, 30000); // 30 second timeout

      this.requestQueue.set(requestId, (response) => {
        clearTimeout(timeout);
        resolve(response);
      });

      this.pythonProcess!.stdin?.write(JSON.stringify(fullRequest) + '\n');
    });
  }

  async search(query: string, k: number = 10, threshold: number = 0.0): Promise<SearchResult[]> {
    const response = await this.sendRequest({
      action: 'search',
      query,
      k,
      threshold
    });

    if (!response.success) {
      throw new Error(response.error || 'Search failed');
    }

    return response.results || [];
  }

  async rebuildIndex(): Promise<string> {
    const response = await this.sendRequest({
      action: 'rebuild_index'
    });

    if (!response.success) {
      throw new Error(response.error || 'Index rebuild failed');
    }

    return response.message || 'Index rebuilt';
  }

  async ping(): Promise<boolean> {
    try {
      const response = await this.sendRequest({
        action: 'ping'
      });
      return response.success;
    } catch {
      return false;
    }
  }
}

// Helper function to format search results for display
export function formatSearchResults(results: SearchResult[]): string {
  if (results.length === 0) {
    return 'No results found.';
  }

  const formatted = results.map((result, index) => {
    const score = (result.score * 100).toFixed(1);
    const confidence = result.confidence ? ` (confidence: ${(result.confidence * 100).toFixed(0)}%)` : '';
    const timestamp = result.timestamp ? ` [${new Date(result.timestamp).toISOString()}]` : '';
    
    return `${index + 1}. ${result.entity_name} (${result.entity_type}) - Score: ${score}%${confidence}${timestamp}
   ${result.observation}`;
  }).join('\n\n');

  return formatted;
}