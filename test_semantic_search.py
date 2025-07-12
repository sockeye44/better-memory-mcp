#!/usr/bin/env python3
"""
Test script for semantic search functionality
"""

import json
import asyncio
from semantic_search import SemanticSearchService

async def test_semantic_search():
    """Test the semantic search service"""
    
    print("Testing Semantic Search Service")
    print("==============================")
    
    # Initialize service
    service = SemanticSearchService("memory.json")
    
    # Load model
    print("\n1. Loading model...")
    service.engine.load_model()
    print("✓ Model loaded")
    
    # Build index
    print("\n2. Building index from memory.json...")
    service.check_and_update_index()
    print(f"✓ Index built with {len(service.engine.metadata)} observations")
    
    # Test queries
    test_queries = [
        "video analysis and temporal reasoning",
        "challenges with team collaboration",
        "AdamW tuning techniques",
        "content safety and RLHF",
    ]
    
    print("\n3. Testing search queries:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        request = {
            'action': 'search',
            'query': query,
            'k': 3,
            'threshold': 0.0
        }
        
        response = await service.handle_request(request)
        
        if response['success']:
            results = response['results']
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results[:3]):
                    score_pct = result['score'] * 100
                    print(f"  {i+1}. [{score_pct:.1f}%] {result['entity_name']} ({result['entity_type']})")
                    print(f"     {result['observation'][:100]}...")
            else:
                print("  No results found")
        else:
            print(f"  Error: {response.get('error', 'Unknown error')}")
    
    print("\n✓ All tests completed")

if __name__ == '__main__':
    asyncio.run(test_semantic_search())