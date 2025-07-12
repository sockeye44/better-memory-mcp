#!/usr/bin/env node

/**
 * Better Memory MCP Server
 * 
 * An enhanced knowledge graph memory system for Claude with temporal tracking,
 * confidence scores, entity archiving, and advanced management capabilities.
 * 
 * Original implementation by Anthropic, PBC
 * Enhanced by @sockeye44 and Claude Opus
 * 
 * Version: 0.8.0
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SemanticSearchBridge, formatSearchResults } from './semantic_search_bridge.js';

// Define memory file path using environment variable with fallback
const defaultMemoryPath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'memory.json');

// If MEMORY_FILE_PATH is just a filename, put it in the same directory as the script
const MEMORY_FILE_PATH = process.env.MEMORY_FILE_PATH
  ? path.isAbsolute(process.env.MEMORY_FILE_PATH)
    ? process.env.MEMORY_FILE_PATH
    : path.join(path.dirname(fileURLToPath(import.meta.url)), process.env.MEMORY_FILE_PATH)
  : defaultMemoryPath;

// We are storing our memory using entities, relations, and observations in a graph structure
interface Observation {
  content: string;
  timestamp: number;
  confidence?: number;
}

interface Entity {
  name: string;
  entityType: string;
  observations: string[] | Observation[];
  createdAt?: number;
  archived?: boolean;
  merged?: boolean;
  mergedInto?: string;
  mergedAt?: number;
}

interface Relation {
  from: string;
  to: string;
  relationType: string;
  createdAt?: number;
}

interface KnowledgeGraph {
  entities: Entity[];
  relations: Relation[];
}

// Compact entity representation for summaries
interface EntitySummary {
  name: string;
  type: string;
  observationCount: number;
  firstObservation?: string;
  relationCount?: number;
}

interface GraphSummary {
  entities: EntitySummary[];
  relations: Relation[];
  totalObservations: number;
}

// Helper functions for observation handling
function isObservationArray(obs: string[] | Observation[]): obs is Observation[] {
  return obs.length > 0 && typeof obs[0] === 'object' && 'content' in obs[0];
}

function normalizeObservations(obs: string[] | Observation[], forceCurrentTimestamp: boolean = false): Observation[] {
  const now = Date.now();
  
  if (isObservationArray(obs)) {
    // If forcing current timestamp, update all observations
    if (forceCurrentTimestamp) {
      return obs.map(o => ({
        ...o,
        timestamp: now
      }));
    }
    return obs;
  }
  
  // Convert legacy string observations to new format with current timestamp
  return obs.map(content => ({
    content,
    timestamp: now,
    confidence: 1.0
  }));
}

function getObservationContents(obs: string[] | Observation[]): string[] {
  if (isObservationArray(obs)) {
    return obs.map(o => o.content);
  }
  return obs;
}

// The KnowledgeGraphManager class contains all operations to interact with the knowledge graph
class KnowledgeGraphManager {
  private async loadGraph(): Promise<KnowledgeGraph> {
    try {
      const data = await fs.readFile(MEMORY_FILE_PATH, "utf-8");
      const lines = data.split("\n").filter(line => line.trim() !== "");
      const now = Date.now();
      
      return lines.reduce((graph: KnowledgeGraph, line) => {
        const item = JSON.parse(line);
        if (item.type === "entity") {
          const entity = item as Entity;
          // Ensure legacy entities have timestamps
          if (!entity.createdAt) {
            entity.createdAt = now;
          }
          // Normalize observations to ensure they have timestamps
          if (entity.observations && entity.observations.length > 0) {
            entity.observations = normalizeObservations(entity.observations);
          }
          graph.entities.push(entity);
        }
        if (item.type === "relation") {
          const relation = item as Relation;
          // Ensure legacy relations have timestamps
          if (!relation.createdAt) {
            relation.createdAt = now;
          }
          graph.relations.push(relation);
        }
        return graph;
      }, { entities: [], relations: [] });
    } catch (error) {
      if (error instanceof Error && 'code' in error && (error as any).code === "ENOENT") {
        return { entities: [], relations: [] };
      }
      throw error;
    }
  }

  private async saveGraph(graph: KnowledgeGraph): Promise<void> {
    const lines = [
      ...graph.entities.map(e => JSON.stringify({ type: "entity", ...e })),
      ...graph.relations.map(r => JSON.stringify({ type: "relation", ...r })),
    ];
    await fs.writeFile(MEMORY_FILE_PATH, lines.join("\n"));
  }

  private createEntitySummary(entity: Entity, relations: Relation[]): EntitySummary {
    const relCount = relations.filter(r => r.from === entity.name || r.to === entity.name).length;
    const contents = getObservationContents(entity.observations);
    return {
      name: entity.name,
      type: entity.entityType,
      observationCount: entity.observations.length,
      firstObservation: contents[0],
      relationCount: relCount
    };
  }

  private createGraphSummary(graph: KnowledgeGraph): GraphSummary {
    const totalObs = graph.entities.reduce((sum, e) => sum + e.observations.length, 0);
    return {
      entities: graph.entities.map(e => this.createEntitySummary(e, graph.relations)),
      relations: graph.relations,
      totalObservations: totalObs
    };
  }

  async createEntities(entities: Entity[]): Promise<{ created: string[], skipped: string[] }> {
    const graph = await this.loadGraph();
    const created: string[] = [];
    const skipped: string[] = [];
    const now = Date.now();
    
    entities.forEach(e => {
      if (!graph.entities.some(existingEntity => existingEntity.name === e.name)) {
        const newEntity: Entity = {
          ...e,
          observations: normalizeObservations(e.observations),
          createdAt: e.createdAt || now
        };
        graph.entities.push(newEntity);
        created.push(e.name);
      } else {
        skipped.push(e.name);
      }
    });
    
    await this.saveGraph(graph);
    return { created, skipped };
  }

  async createRelations(relations: Relation[]): Promise<{ created: number, skipped: number }> {
    const graph = await this.loadGraph();
    let created = 0;
    let skipped = 0;
    const now = Date.now();
    
    relations.forEach(r => {
      if (!graph.relations.some(existingRelation => 
        existingRelation.from === r.from && 
        existingRelation.to === r.to && 
        existingRelation.relationType === r.relationType
      )) {
        const newRelation: Relation = {
          ...r,
          createdAt: r.createdAt || now
        };
        graph.relations.push(newRelation);
        created++;
      } else {
        skipped++;
      }
    });
    
    await this.saveGraph(graph);
    return { created, skipped };
  }

  async addObservations(observations: { entityName: string; contents: string[]; confidence?: number[] }[]): Promise<{ entityName: string; added: number; skipped: number }[]> {
    const graph = await this.loadGraph();
    const now = Date.now();
    
    const results = observations.map(o => {
      const entity = graph.entities.find(e => e.name === o.entityName);
      if (!entity) {
        throw new Error(`Entity '${o.entityName}' not found`);
      }
      
      // Normalize existing observations
      const normalizedObs = normalizeObservations(entity.observations);
      const existingContents = normalizedObs.map(obs => obs.content);
      
      // Filter new observations
      const newContents = o.contents.filter(content => !existingContents.includes(content));
      
      // Create new observation objects
      const newObservations: Observation[] = newContents.map((content, index) => ({
        content,
        timestamp: now,
        confidence: o.confidence?.[index] ?? 1.0
      }));
      
      // Update entity with all observations
      entity.observations = [...normalizedObs, ...newObservations];
      
      return { 
        entityName: o.entityName, 
        added: newObservations.length,
        skipped: o.contents.length - newContents.length
      };
    });
    
    await this.saveGraph(graph);
    return results;
  }

  async deleteEntities(entityNames: string[]): Promise<{ deleted: number, cascadedRelations: number }> {
    const graph = await this.loadGraph();
    const beforeCount = graph.entities.length;
    const beforeRelCount = graph.relations.length;
    
    graph.entities = graph.entities.filter(e => !entityNames.includes(e.name));
    graph.relations = graph.relations.filter(r => !entityNames.includes(r.from) && !entityNames.includes(r.to));
    
    await this.saveGraph(graph);
    return {
      deleted: beforeCount - graph.entities.length,
      cascadedRelations: beforeRelCount - graph.relations.length
    };
  }

  async deleteObservations(deletions: { entityName: string; observations: string[] }[]): Promise<{ entityName: string; deleted: number }[]> {
    const graph = await this.loadGraph();
    const results = deletions.map(d => {
      const entity = graph.entities.find(e => e.name === d.entityName);
      if (!entity) return { entityName: d.entityName, deleted: 0 };
      
      const beforeCount = entity.observations.length;
      
      if (isObservationArray(entity.observations)) {
        // Handle new format
        entity.observations = entity.observations.filter(o => !d.observations.includes(o.content));
      } else {
        // Handle legacy format
        entity.observations = entity.observations.filter(o => !d.observations.includes(o));
      }
      
      return { entityName: d.entityName, deleted: beforeCount - entity.observations.length };
    });
    await this.saveGraph(graph);
    return results;
  }

  async deleteRelations(relations: Relation[]): Promise<{ deleted: number }> {
    const graph = await this.loadGraph();
    const beforeCount = graph.relations.length;
    
    graph.relations = graph.relations.filter(r => !relations.some(delRelation => 
      r.from === delRelation.from && 
      r.to === delRelation.to && 
      r.relationType === delRelation.relationType
    ));
    
    await this.saveGraph(graph);
    return { deleted: beforeCount - graph.relations.length };
  }

  async readGraph(detailLevel: string = "summary", entityNames?: string[], includeArchived: boolean = false, includeMerged: boolean = false): Promise<KnowledgeGraph | GraphSummary> {
    const graph = await this.loadGraph();
    
    // Filter out archived and merged entities unless specifically requested
    let entities = graph.entities.filter(e => {
      if (!includeArchived && e.archived) return false;
      if (!includeMerged && e.merged) return false;
      return true;
    });
    
    if (entityNames && entityNames.length > 0) {
      // Return specific entities with full details
      entities = entities.filter(e => entityNames.includes(e.name));
      const entityNameSet = new Set(entities.map(e => e.name));
      const filteredRelations = graph.relations.filter(r => 
        entityNameSet.has(r.from) && entityNameSet.has(r.to)
      );
      return { entities, relations: filteredRelations };
    }
    
    const entityNameSet = new Set(entities.map(e => e.name));
    const activeRelations = graph.relations.filter(r => 
      entityNameSet.has(r.from) && entityNameSet.has(r.to)
    );
    
    switch (detailLevel) {
      case "minimal":
        // Just names and counts
        return {
          entities: entities.map(e => ({
            name: e.name,
            type: e.entityType,
            observationCount: e.observations.length
          })),
          relations: activeRelations,
          totalObservations: entities.reduce((sum, e) => sum + e.observations.length, 0)
        };
      
      case "full":
        return { entities, relations: activeRelations };
      
      case "summary":
      default:
        return this.createGraphSummary({ entities, relations: activeRelations });
    }
  }

  async getEntityDetails(entityNames: string[]): Promise<Entity[]> {
    const graph = await this.loadGraph();
    return graph.entities.filter(e => entityNames.includes(e.name));
  }

  async searchNodes(query: string, maxObservations: number = 3): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph();
    const queryLower = query.toLowerCase();
    
    // Filter entities (exclude archived and merged by default)
    const filteredEntities = graph.entities.filter(e => {
      if (e.archived || e.merged) return false;
      
      const contents = getObservationContents(e.observations);
      return e.name.toLowerCase().includes(queryLower) ||
        e.entityType.toLowerCase().includes(queryLower) ||
        contents.some(o => o.toLowerCase().includes(queryLower));
    }).map(e => ({
      ...e,
      observations: e.observations.slice(0, maxObservations)
    }));
  
    // Create a Set of filtered entity names for quick lookup
    const filteredEntityNames = new Set(filteredEntities.map(e => e.name));
  
    // Filter relations to only include those between filtered entities
    const filteredRelations = graph.relations.filter(r => 
      filteredEntityNames.has(r.from) && filteredEntityNames.has(r.to)
    );
  
    return {
      entities: filteredEntities,
      relations: filteredRelations,
    };
  }

  async semanticSearch(query: string, k: number = 10, threshold: number = 0.0): Promise<any> {
    try {
      // Check if semantic search is available
      const isAvailable = await semanticSearchBridge.ping();
      if (!isAvailable) {
        // Fall back to regular search
        console.error('Semantic search unavailable, falling back to keyword search');
        return this.searchNodes(query, k);
      }

      // Perform semantic search
      const results = await semanticSearchBridge.search(query, k, threshold);
      
      // Group results by entity
      const entityMap = new Map<string, any>();
      const entityScores = new Map<string, number>();
      
      for (const result of results) {
        if (!entityMap.has(result.entity_name)) {
          // Get full entity details
          const [entity] = await this.getEntityDetails([result.entity_name]);
          if (entity) {
            entityMap.set(result.entity_name, entity);
            entityScores.set(result.entity_name, result.score);
          }
        } else {
          // Update score if this result has a higher score
          const currentScore = entityScores.get(result.entity_name) || 0;
          if (result.score > currentScore) {
            entityScores.set(result.entity_name, result.score);
          }
        }
      }
      
      // Sort entities by score
      const sortedEntities = Array.from(entityMap.values())
        .sort((a, b) => (entityScores.get(b.name) || 0) - (entityScores.get(a.name) || 0))
        .slice(0, k);
      
      // Get relations between matched entities
      const entityNames = new Set(sortedEntities.map(e => e.name));
      const graph = await this.loadGraph();
      const relations = graph.relations.filter(r => 
        entityNames.has(r.from) && entityNames.has(r.to)
      );
      
      return {
        entities: sortedEntities,
        relations,
        searchResults: results,
        searchType: 'semantic'
      };
    } catch (error) {
      console.error('Semantic search error:', error);
      // Fall back to regular search
      const result = await this.searchNodes(query, k);
      return { ...result, searchType: 'keyword' };
    }
  }

  async openNodes(names: string[]): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph();
    
    // Filter entities
    const filteredEntities = graph.entities.filter(e => names.includes(e.name));
  
    // Create a Set of filtered entity names for quick lookup
    const filteredEntityNames = new Set(filteredEntities.map(e => e.name));
  
    // Filter relations to only include those between filtered entities
    const filteredRelations = graph.relations.filter(r => 
      filteredEntityNames.has(r.from) && filteredEntityNames.has(r.to)
    );
  
    return {
      entities: filteredEntities,
      relations: filteredRelations,
    };
  }

  async archiveEntity(entityName: string): Promise<{ success: boolean, message: string }> {
    const graph = await this.loadGraph();
    const entity = graph.entities.find(e => e.name === entityName);
    
    if (!entity) {
      return { success: false, message: `Entity '${entityName}' not found` };
    }
    
    if (entity.archived) {
      return { success: false, message: `Entity '${entityName}' is already archived` };
    }
    
    entity.archived = true;
    await this.saveGraph(graph);
    
    return { success: true, message: `Successfully archived entity '${entityName}'` };
  }

  async unarchiveEntity(entityName: string): Promise<{ success: boolean, message: string }> {
    const graph = await this.loadGraph();
    const entity = graph.entities.find(e => e.name === entityName);
    
    if (!entity) {
      return { success: false, message: `Entity '${entityName}' not found` };
    }
    
    if (!entity.archived) {
      return { success: false, message: `Entity '${entityName}' is not archived` };
    }
    
    entity.archived = false;
    await this.saveGraph(graph);
    
    return { success: true, message: `Successfully unarchived entity '${entityName}'` };
  }

  async mergeEntities(sourceName: string, targetName: string): Promise<{ success: boolean, message: string }> {
    const graph = await this.loadGraph();
    const now = Date.now();
    
    const sourceEntity = graph.entities.find(e => e.name === sourceName);
    const targetEntity = graph.entities.find(e => e.name === targetName);
    
    if (!sourceEntity) {
      return { success: false, message: `Source entity '${sourceName}' not found` };
    }
    
    if (!targetEntity) {
      return { success: false, message: `Target entity '${targetName}' not found` };
    }
    
    if (sourceEntity.merged) {
      return { success: false, message: `Source entity '${sourceName}' has already been merged` };
    }
    
    // Merge observations
    const sourceObs = normalizeObservations(sourceEntity.observations);
    const targetObs = normalizeObservations(targetEntity.observations);
    const targetContents = targetObs.map(o => o.content);
    
    // Add non-duplicate observations from source to target
    const newObservations = sourceObs.filter(o => !targetContents.includes(o.content));
    targetEntity.observations = [...targetObs, ...newObservations];
    
    // Update relations - redirect all source relations to target
    graph.relations = graph.relations.map(r => {
      if (r.from === sourceName) {
        return { ...r, from: targetName };
      }
      if (r.to === sourceName) {
        return { ...r, to: targetName };
      }
      return r;
    });
    
    // Remove duplicate relations that may have been created
    const uniqueRelations: Relation[] = [];
    graph.relations.forEach(r => {
      if (!uniqueRelations.some(ur => 
        ur.from === r.from && 
        ur.to === r.to && 
        ur.relationType === r.relationType
      )) {
        uniqueRelations.push(r);
      }
    });
    graph.relations = uniqueRelations;
    
    // Soft-delete source entity by marking it as merged
    sourceEntity.merged = true;
    sourceEntity.mergedInto = targetName;
    sourceEntity.mergedAt = now;
    
    await this.saveGraph(graph);
    
    return { 
      success: true, 
      message: `Successfully merged '${sourceName}' into '${targetName}'. Merged ${newObservations.length} observations and updated ${graph.relations.filter(r => r.from === targetName || r.to === targetName).length} relations.`
    };
  }

  async getRecentChanges(hours: number = 24): Promise<{
    recentEntities: Entity[],
    recentRelations: Relation[],
    recentObservations: { entity: string, observations: Observation[] }[]
  }> {
    const graph = await this.loadGraph();
    const cutoffTime = Date.now() - (hours * 60 * 60 * 1000);
    
    // Find recently created entities (exclude archived and merged)
    const recentEntities = graph.entities.filter(e => 
      e.createdAt && e.createdAt >= cutoffTime && !e.archived && !e.merged
    );
    
    // Find recently created relations
    const recentRelations = graph.relations.filter(r => 
      r.createdAt && r.createdAt >= cutoffTime
    );
    
    // Find entities with recent observations
    const recentObservations: { entity: string, observations: Observation[] }[] = [];
    
    graph.entities.forEach(entity => {
      if (!entity.archived && !entity.merged) {
        const normalizedObs = normalizeObservations(entity.observations);
        const recent = normalizedObs.filter(o => o.timestamp >= cutoffTime);
        
        if (recent.length > 0) {
          recentObservations.push({
            entity: entity.name,
            observations: recent
          });
        }
      }
    });
    
    return {
      recentEntities,
      recentRelations,
      recentObservations
    };
  }
}

const knowledgeGraphManager = new KnowledgeGraphManager();

// Initialize semantic search bridge
const semanticSearchBridge = new SemanticSearchBridge(MEMORY_FILE_PATH);

// Start semantic search service in the background
semanticSearchBridge.start().catch(error => {
  console.error('Warning: Semantic search service failed to start:', error.message);
  console.error('Semantic search will be unavailable, but other features will work normally.');
});

// The server instance and tools exposed to Claude
const server = new Server({
  name: "better-memory-mcp",
  version: "0.8.0",
}, {
  capabilities: {
    tools: {},
  },
});

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "create_entities",
        description: "Create new entities in the knowledge graph",
        inputSchema: {
          type: "object",
          properties: {
            entities: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  name: { type: "string", description: "Entity identifier" },
                  entityType: { type: "string", description: "Entity type" },
                  observations: { 
                    type: "array", 
                    items: { type: "string" },
                    description: "Observations about the entity"
                  },
                },
                required: ["name", "entityType", "observations"],
              },
            },
          },
          required: ["entities"],
        },
      },
      {
        name: "create_relations",
        description: "Create relations between entities (active voice)",
        inputSchema: {
          type: "object",
          properties: {
            relations: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  from: { type: "string", description: "Source entity" },
                  to: { type: "string", description: "Target entity" },
                  relationType: { type: "string", description: "Relation type" },
                },
                required: ["from", "to", "relationType"],
              },
            },
          },
          required: ["relations"],
        },
      },
      {
        name: "add_observations",
        description: "Add observations to existing entities",
        inputSchema: {
          type: "object",
          properties: {
            observations: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  entityName: { type: "string", description: "Target entity" },
                  contents: { 
                    type: "array", 
                    items: { type: "string" },
                    description: "New observations"
                  },
                  confidence: {
                    type: "array",
                    items: { type: "number" },
                    description: "Confidence scores (0-1) for each observation (optional)"
                  },
                },
                required: ["entityName", "contents"],
              },
            },
          },
          required: ["observations"],
        },
      },
      {
        name: "delete_entities",
        description: "Delete entities and their relations",
        inputSchema: {
          type: "object",
          properties: {
            entityNames: { 
              type: "array", 
              items: { type: "string" },
              description: "Entity names to delete" 
            },
          },
          required: ["entityNames"],
        },
      },
      {
        name: "delete_observations",
        description: "Delete specific observations from entities",
        inputSchema: {
          type: "object",
          properties: {
            deletions: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  entityName: { type: "string" },
                  observations: { 
                    type: "array", 
                    items: { type: "string" }
                  },
                },
                required: ["entityName", "observations"],
              },
            },
          },
          required: ["deletions"],
        },
      },
      {
        name: "delete_relations",
        description: "Delete specific relations",
        inputSchema: {
          type: "object",
          properties: {
            relations: { 
              type: "array", 
              items: {
                type: "object",
                properties: {
                  from: { type: "string" },
                  to: { type: "string" },
                  relationType: { type: "string" },
                },
                required: ["from", "to", "relationType"],
              }
            },
          },
          required: ["relations"],
        },
      },
      {
        name: "read_graph",
        description: "Read graph with controllable detail. Default returns summary with entity names, types, and first observation only.",
        inputSchema: {
          type: "object",
          properties: {
            detailLevel: { 
              type: "string", 
              enum: ["minimal", "summary", "full"],
              description: "Level of detail: minimal (names+counts), summary (default, includes first observation), full (everything)",
              default: "summary"
            },
            entityNames: {
              type: "array",
              items: { type: "string" },
              description: "Optional: Get full details for specific entities only"
            },
            includeArchived: {
              type: "boolean",
              description: "Include archived entities (default false)",
              default: false
            },
            includeMerged: {
              type: "boolean",
              description: "Include merged entities (default false)",
              default: false
            }
          },
        },
      },
      {
        name: "get_entity_details",
        description: "Get complete details including all observations for specific entities",
        inputSchema: {
          type: "object",
          properties: {
            entityNames: {
              type: "array",
              items: { type: "string" },
              description: "Entity names to get full details for"
            }
          },
          required: ["entityNames"],
        },
      },
      {
        name: "search_nodes",
        description: "Search for entities strictly matching a query",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "Search query" },
            maxObservations: { 
              type: "number", 
              description: "Max observations per entity (default 5)",
              default: 5
            }
          },
          required: ["query"],
        },
      },
      {
        name: "semantic_search",
        description: "Advanced semantic search using ModernColBERT embeddings (recommended!). Understands meaning and context, not just keywords. Returns entities with highest semantic similarity to the query.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "Natural language search query" },
            k: { 
              type: "number", 
              description: "Number of results to return (default 10)",
              default: 10
            },
            threshold: {
              type: "number",
              description: "Minimum similarity score (0-1, default 0)",
              default: 0
            }
          },
          required: ["query"],
        },
      },
      {
        name: "open_nodes",
        description: "Open specific nodes by name with full details",
        inputSchema: {
          type: "object",
          properties: {
            names: {
              type: "array",
              items: { type: "string" },
              description: "Entity names to retrieve",
            },
          },
          required: ["names"],
        },
      },
      {
        name: "merge_entities",
        description: "Merge one entity into another, combining observations and updating relations",
        inputSchema: {
          type: "object",
          properties: {
            sourceName: { type: "string", description: "Entity to merge from (will be deleted)" },
            targetName: { type: "string", description: "Entity to merge into (will be preserved)" },
          },
          required: ["sourceName", "targetName"],
        },
      },
      {
        name: "archive_entity",
        description: "Archive an entity (soft delete - hidden from normal queries)",
        inputSchema: {
          type: "object",
          properties: {
            entityName: { type: "string", description: "Entity to archive" },
          },
          required: ["entityName"],
        },
      },
      {
        name: "unarchive_entity",
        description: "Unarchive a previously archived entity",
        inputSchema: {
          type: "object",
          properties: {
            entityName: { type: "string", description: "Entity to unarchive" },
          },
          required: ["entityName"],
        },
      },
      {
        name: "get_recent_changes",
        description: "Get entities, relations, and observations created/modified within specified hours",
        inputSchema: {
          type: "object",
          properties: {
            hours: { 
              type: "number", 
              description: "Number of hours to look back (default 24)",
              default: 24
            },
          },
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (!args) {
    throw new Error(`No arguments provided for tool: ${name}`);
  }

  try {
    switch (name) {
      case "create_entities":
        const createResult = await knowledgeGraphManager.createEntities(args.entities as Entity[]);
        return { content: [{ type: "text", text: JSON.stringify(createResult) }] };
      
      case "create_relations":
        const relResult = await knowledgeGraphManager.createRelations(args.relations as Relation[]);
        return { content: [{ type: "text", text: JSON.stringify(relResult) }] };
      
      case "add_observations":
        const obsResult = await knowledgeGraphManager.addObservations(args.observations as { entityName: string; contents: string[]; confidence?: number[] }[]);
        return { content: [{ type: "text", text: JSON.stringify(obsResult) }] };
      
      case "delete_entities":
        const delEntResult = await knowledgeGraphManager.deleteEntities(args.entityNames as string[]);
        return { content: [{ type: "text", text: JSON.stringify(delEntResult) }] };
      
      case "delete_observations":
        const delObsResult = await knowledgeGraphManager.deleteObservations(args.deletions as { entityName: string; observations: string[] }[]);
        return { content: [{ type: "text", text: JSON.stringify(delObsResult) }] };
      
      case "delete_relations":
        const delRelResult = await knowledgeGraphManager.deleteRelations(args.relations as Relation[]);
        return { content: [{ type: "text", text: JSON.stringify(delRelResult) }] };
      
      case "read_graph":
        const graphResult = await knowledgeGraphManager.readGraph(
          args.detailLevel as string || "summary",
          args.entityNames as string[] | undefined,
          args.includeArchived as boolean || false,
          args.includeMerged as boolean || false
        );
        return { content: [{ type: "text", text: JSON.stringify(graphResult) }] };
      
      case "get_entity_details":
        const detailsResult = await knowledgeGraphManager.getEntityDetails(args.entityNames as string[]);
        return { content: [{ type: "text", text: JSON.stringify(detailsResult) }] };
      
      case "search_nodes":
        const searchResult = await knowledgeGraphManager.searchNodes(
          args.query as string,
          args.maxObservations as number || 3
        );
        return { content: [{ type: "text", text: JSON.stringify(searchResult) }] };
      
      case "semantic_search":
        const semanticResult = await knowledgeGraphManager.semanticSearch(
          args.query as string,
          args.k as number || 10,
          args.threshold as number || 0
        );
        return { content: [{ type: "text", text: JSON.stringify(semanticResult) }] };
      
      case "open_nodes":
        const openResult = await knowledgeGraphManager.openNodes(args.names as string[]);
        return { content: [{ type: "text", text: JSON.stringify(openResult) }] };
      
      case "merge_entities":
        const mergeResult = await knowledgeGraphManager.mergeEntities(
          args.sourceName as string,
          args.targetName as string
        );
        return { content: [{ type: "text", text: JSON.stringify(mergeResult) }] };
      
      case "archive_entity":
        const archiveResult = await knowledgeGraphManager.archiveEntity(args.entityName as string);
        return { content: [{ type: "text", text: JSON.stringify(archiveResult) }] };
      
      case "unarchive_entity":
        const unarchiveResult = await knowledgeGraphManager.unarchiveEntity(args.entityName as string);
        return { content: [{ type: "text", text: JSON.stringify(unarchiveResult) }] };
      
      case "get_recent_changes":
        const recentResult = await knowledgeGraphManager.getRecentChanges(args.hours as number || 24);
        return { content: [{ type: "text", text: JSON.stringify(recentResult) }] };
      
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    // Return error in a format that won't break the response
    return { 
      content: [{ 
        type: "text", 
        text: JSON.stringify({ 
          error: error instanceof Error ? error.message : "Unknown error",
          tool: name 
        }) 
      }] 
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Better Memory MCP Server v0.8.0 running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
