#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

from pdf_chunker.ai_enrichment import _load_tag_configs, classify_chunk_utterance

def test_ai_enrichment():
    """Test enhanced AI enrichment with tag configuration"""
    
    # Test tag configuration loading
    print('Testing tag configuration loading...')
    try:
        tag_configs = _load_tag_configs()
        
        if tag_configs:
            print(f'  Successfully loaded {len(tag_configs)} tag categories:')
            for category, tags in tag_configs.items():
                print(f'    {category}: {len(tags)} tags')
                if len(tags) > 0:
                    print(f'      Sample tags: {tags[:3]}')
        else:
            print('  WARNING: No tag configurations loaded')
            return False
            
        # Test AI enrichment with a sample text chunk
        print()
        print('Testing AI enrichment with sample text...')
        
        sample_texts = [
            'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
            'The project manager should create a detailed timeline with milestones and deliverables to ensure successful project completion.',
            'Consciousness is the state of being aware of and able to think about one\'s existence, sensations, thoughts, and surroundings.'
        ]
        
        success_count = 0
        for i, text in enumerate(sample_texts, 1):
            print(f'  Sample {i}: Testing classification and tagging...')
            try:
                result = classify_chunk_utterance(text, tag_configs)
                
                if result:
                    classification = result.get('classification', 'unknown')
                    tags = result.get('tags', [])
                    
                    print(f'    Classification: {classification}')
                    print(f'    Tags: {tags}')
                    
                    # Validate tags against available vocabulary
                    all_valid_tags = set()
                    for category_tags in tag_configs.values():
                        all_valid_tags.update(category_tags)
                    
                    invalid_tags = [tag for tag in tags if tag not in all_valid_tags]
                    if invalid_tags:
                        print(f'    WARNING: Invalid tags found: {invalid_tags}')
                    else:
                        print(f'    All tags are valid')
                        success_count += 1
                else:
                    print(f'    ERROR: No result returned from AI enrichment')
                    
            except Exception as e:
                print(f'    ERROR: AI enrichment failed: {e}')
            
            print()
            
        return success_count > 0
        
    except Exception as e:
        print(f'ERROR: Tag configuration testing failed: {e}')
        return False

if __name__ == '__main__':
    success = test_ai_enrichment()
    sys.exit(0 if success else 1)
