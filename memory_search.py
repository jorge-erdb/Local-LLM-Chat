import json
import os
import re
import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class MemorySearch:
    def __init__(self, memory_dir='./memory'):
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, 'memory.jsonl')
        self.index_file = os.path.join(memory_dir, 'index.json')
        
        # Load search index if it exists
        self.index = self.load_index()

    def load_index(self) -> dict:
        """Load the search index"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading search index: {e}")
        
        return {
            'vocabulary': {},
            'entities': {},
            'topics': {},
            'content_types': {},
            'sessions': {},
            'timestamps': {}
        }

    def normalize_query(self, query: str) -> List[str]:
        """Normalize and tokenize search query"""
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', query.lower())
        # Split into words and filter out empty strings
        words = [word.strip() for word in normalized.split() if word.strip()]
        return words

    def get_synonyms_and_variations(self, word: str) -> List[str]:
        """Get synonyms and variations for a word (enhanced version)"""
        variations = {
            'name': ['called', 'named', 'names', 'title', 'nickname'],
            'called': ['name', 'named', 'names', 'known'],
            'remember': ['recall', 'recollect', 'remind', 'memorize', 'memory'],
            'recall': ['remember', 'recollect', 'remind', 'memory'],
            'said': ['told', 'mentioned', 'stated', 'spoke', 'talked'],
            'told': ['said', 'mentioned', 'informed', 'spoke', 'talked'],
            'story': ['tale', 'narrative', 'chapter', 'book', 'novel'],
            'chapter': ['part', 'section', 'story', 'episode'],
            'favorite': ['favourite', 'preferred', 'liked', 'best'],
            'color': ['colour', 'shade', 'hue'],
            'like': ['enjoy', 'love', 'prefer', 'fancy'],
            'write': ['writing', 'wrote', 'written', 'author'],
            'coding': ['programming', 'code', 'development'],
            'help': ['assist', 'support', 'aid', 'guide'],
            'problem': ['issue', 'trouble', 'error', 'bug']
        }
        
        word_lower = word.lower()
        result = [word_lower]
        
        # Add direct variations
        if word_lower in variations:
            result.extend(variations[word_lower])
        
        # Add plurals/singulars
        if word_lower.endswith('s') and len(word_lower) > 3:
            result.append(word_lower[:-1])  # Remove 's'
        elif not word_lower.endswith('s'):
            result.append(word_lower + 's')  # Add 's'
        
        # Add common suffixes variations
        if word_lower.endswith('ing'):
            base = word_lower[:-3]
            result.extend([base, base + 'ed'])
        elif word_lower.endswith('ed'):
            base = word_lower[:-2]
            result.extend([base, base + 'ing'])
        
        return list(set(result))

    def fuzzy_match_score(self, word1: str, word2: str) -> float:
        """Calculate fuzzy match score between two words using Levenshtein distance"""
        if len(word1) == 0 or len(word2) == 0:
            return 0.0
        
        # Levenshtein distance calculation
        d = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        
        for i in range(len(word1) + 1):
            d[i][0] = i
        for j in range(len(word2) + 1):
            d[0][j] = j
        
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                cost = 0 if word1[i-1] == word2[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion  
                    d[i-1][j-1] + cost  # substitution
                )
        
        # Calculate similarity score (0-1)
        max_len = max(len(word1), len(word2))
        similarity = 1 - (d[len(word1)][len(word2)] / max_len)
        return similarity

    def get_candidate_message_ids(self, query_words: List[str]) -> Set[str]:
        """Get candidate message IDs using index lookup"""
        candidate_ids = set()
        
        # Search in vocabulary index
        for word in query_words:
            # Direct matches
            if word in self.index.get('vocabulary', {}):
                candidate_ids.update(self.index['vocabulary'][word])
            
            # Synonym matches
            for variant in self.get_synonyms_and_variations(word):
                if variant in self.index.get('vocabulary', {}):
                    candidate_ids.update(self.index['vocabulary'][variant])
            
            # Entity matches
            if word.capitalize() in self.index.get('entities', {}):
                candidate_ids.update(self.index['entities'][word.capitalize()])
        
        return candidate_ids

    def load_memory_entries(self, message_ids: Set[str] = None) -> Dict[str, dict]:
        """Load memory entries, optionally filtered by message IDs"""
        entries = {}
        
        if not os.path.exists(self.memory_file):
            return entries
        
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_id = entry.get('id')
                    
                    # If filtering by IDs and this ID is not in the set, skip
                    if message_ids is not None and entry_id not in message_ids:
                        continue
                    
                    entries[entry_id] = entry
                except json.JSONDecodeError:
                    continue
        
        return entries

    def calculate_relevance_score(self, entry: dict, query_words: List[str], query: str) -> float:
        """Calculate enhanced relevance score for a memory entry"""
        score = 0.0
        content = entry.get('content', '').lower()
        keywords = [kw.lower() for kw in entry.get('keywords', [])]
        entities = [ent.lower() for ent in entry.get('entities', [])]
        topics = entry.get('topics', [])
        title = entry.get('conversation_title', '').lower()
        
        # 1. Exact phrase match (highest score)
        if query.lower() in content:
            score += 20
        
        # 2. Exact word matches in content
        content_words = set(re.findall(r'\b\w+\b', content))
        exact_matches = len(set(query_words) & content_words)
        score += exact_matches * 5
        
        # 3. Keyword matches (indexed keywords are important)
        keyword_matches = len(set(query_words) & set(keywords))
        score += keyword_matches * 4
        
        # 4. Entity matches
        entity_matches = len(set(query_words) & set(entities))
        score += entity_matches * 6  # Entities are very important
        
        # 5. Fuzzy matching for typos and variations
        for q_word in query_words:
            best_fuzzy_score = 0
            for c_word in content_words:
                if len(q_word) > 2 and len(c_word) > 2:
                    fuzzy_score = self.fuzzy_match_score(q_word, c_word)
                    best_fuzzy_score = max(best_fuzzy_score, fuzzy_score)
            
            if best_fuzzy_score > 0.7:  # Threshold for fuzzy match
                score += best_fuzzy_score * 3
        
        # 6. Title relevance
        title_words = set(re.findall(r'\b\w+\b', title))
        title_matches = len(set(query_words) & title_words)
        score += title_matches * 3
        
        # 7. Topic relevance
        query_text = ' '.join(query_words)
        topic_bonus = {
            'personal_info': ['name', 'favorite', 'like', 'age'],
            'creative_writing': ['story', 'chapter', 'write'],
            'memory_recall': ['recall', 'remember', 'forgot'],
        }
        for topic in topics:
            if topic in topic_bonus:
                if any(word in query_text for word in topic_bonus[topic]):
                    score += 2
        
        # 8. Importance weighting
        importance = entry.get('importance', 5)
        score += (importance / 10) * 2
        
        # 9. Content type bonuses
        content_type = entry.get('content_type', '')
        if 'question' in query_text and content_type == 'question':
            score += 1
        
        # 10. Recency bonus
        timestamp = entry.get('timestamp', '')
        if timestamp:
            try:
                msg_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                current_date = datetime.now(msg_date.tzinfo) if msg_date.tzinfo else datetime.now()
                days_old = (current_date - msg_date).days
                
                if days_old < 1:
                    score += 3  # Today
                elif days_old < 7:
                    score += 2  # This week  
                elif days_old < 30:
                    score += 1  # This month
            except:
                pass
        
        # 11. Role bonus (user messages often more informative for recall)
        if entry.get('role') == 'user':
            score += 0.5
        
        return score

    def search_memory(self, query: str, max_results: int = 10, min_score: float = 1.0) -> List[dict]:
        """Enhanced search through memory with metadata"""
        if not query.strip():
            return []
        
        query_words = self.normalize_query(query)
        if not query_words:
            return []
        
        # Get candidate message IDs from index (fast filtering)
        candidate_ids = self.get_candidate_message_ids(query_words)
        
        # If no candidates found via index, fall back to full search
        if not candidate_ids:
            memory_entries = self.load_memory_entries()
        else:
            memory_entries = self.load_memory_entries(candidate_ids)
        
        results = []
        
        for entry_id, entry in memory_entries.items():
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(entry, query_words, query)
            
            if relevance_score >= min_score:
                # Format result similar to original search
                result = {
                    'session_id': entry.get('session_id'),
                    'conversation_title': entry.get('conversation_title'),
                    'conversation_date': self.get_session_start_date(entry.get('session_id')),
                    'matched_message': {
                        'role': entry.get('role'),
                        'content': entry.get('content'),
                        'timestamp': entry.get('timestamp')
                    },
                    'context_messages': self.get_context_messages(entry),
                    'relevance_score': relevance_score,
                    'metadata': {
                        'keywords': entry.get('keywords', []),
                        'entities': entry.get('entities', []),
                        'topics': entry.get('topics', []),
                        'importance': entry.get('importance'),
                        'content_type': entry.get('content_type')
                    }
                }
                results.append(result)
        
        # Sort by relevance score (highest first) and limit results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]

    def get_context_messages(self, entry: dict, context_window: int = 2) -> List[dict]:
        """Get context messages around a matched message"""
        session_id = entry.get('session_id')
        message_index = entry.get('message_index')
        
        context_messages = []
        
        # Load other messages from the same session
        session_entries = {}
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    other_entry = json.loads(line.strip())
                    if other_entry.get('session_id') == session_id:
                        session_entries[other_entry.get('message_index')] = other_entry
                except json.JSONDecodeError:
                    continue
        
        # Get messages in context window
        start_idx = max(1, message_index - context_window)
        end_idx = message_index + context_window + 1
        
        for idx in range(start_idx, end_idx):
            if idx in session_entries:
                msg_entry = session_entries[idx]
                context_messages.append({
                    'role': msg_entry.get('role'),
                    'content': self.truncate_content(msg_entry.get('content', ''), 150),
                    'is_match': (idx == message_index)
                })
        
        return context_messages

    def truncate_content(self, content: str, max_length: int = 150) -> str:
        """Truncate content to specified length"""
        if len(content) <= max_length:
            return content
        return content[:max_length] + '...'

    def get_session_start_date(self, session_id: str) -> str:
        """Get the start date of a session from the first message"""
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('session_id') == session_id and entry.get('message_index') == 1:
                        return entry.get('timestamp', '')
                except json.JSONDecodeError:
                    continue
        return ''

    def search_by_topic(self, topic: str, max_results: int = 10) -> List[dict]:
        """Search for messages by topic"""
        if topic not in self.index.get('topics', {}):
            return []
        
        message_ids = set(self.index['topics'][topic])
        memory_entries = self.load_memory_entries(message_ids)
        
        results = []
        for entry_id, entry in memory_entries.items():
            result = {
                'session_id': entry.get('session_id'),
                'conversation_title': entry.get('conversation_title'),
                'matched_message': {
                    'role': entry.get('role'),
                    'content': entry.get('content'),
                    'timestamp': entry.get('timestamp')
                },
                'relevance_score': entry.get('importance', 5),
                'metadata': {
                    'keywords': entry.get('keywords', []),
                    'entities': entry.get('entities', []),
                    'topics': entry.get('topics', []),
                    'importance': entry.get('importance'),
                    'content_type': entry.get('content_type')
                }
            }
            results.append(result)
        
        # Sort by importance and limit
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]

    def search_by_entity(self, entity: str, max_results: int = 10) -> List[dict]:
        """Search for messages containing a specific entity"""
        if entity not in self.index.get('entities', {}):
            return []
        
        message_ids = set(self.index['entities'][entity])
        return self.search_by_ids(message_ids, max_results)

    def search_by_ids(self, message_ids: Set[str], max_results: int = 10) -> List[dict]:
        """Helper method to search by message IDs"""
        memory_entries = self.load_memory_entries(message_ids)
        
        results = []
        for entry_id, entry in memory_entries.items():
            result = {
                'session_id': entry.get('session_id'),
                'conversation_title': entry.get('conversation_title'),
                'matched_message': {
                    'role': entry.get('role'),
                    'content': entry.get('content'),
                    'timestamp': entry.get('timestamp')
                },
                'context_messages': self.get_context_messages(entry),
                'relevance_score': entry.get('importance', 5),
                'metadata': {
                    'keywords': entry.get('keywords', []),
                    'entities': entry.get('entities', []),
                    'topics': entry.get('topics', []),
                    'importance': entry.get('importance'),
                    'content_type': entry.get('content_type')
                }
            }
            results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]

    def get_memory_stats(self) -> dict:
        """Get statistics about the memory database"""
        if not os.path.exists(self.memory_file):
            return {'total_messages': 0, 'total_sessions': 0, 'topics': {}, 'entities_count': 0}
        
        stats = {
            'total_messages': 0,
            'total_sessions': set(),
            'topics': defaultdict(int),
            'content_types': defaultdict(int),
            'entities_count': 0,
            'avg_importance': 0,
            'total_tokens': 0
        }
        
        importance_sum = 0
        
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats['total_messages'] += 1
                    stats['total_sessions'].add(entry.get('session_id'))
                    stats['entities_count'] += len(entry.get('entities', []))
                    stats['total_tokens'] += entry.get('tokens', 0)
                    importance_sum += entry.get('importance', 5)
                    
                    for topic in entry.get('topics', []):
                        stats['topics'][topic] += 1
                    
                    content_type = entry.get('content_type', 'unknown')
                    stats['content_types'][content_type] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        if stats['total_messages'] > 0:
            stats['avg_importance'] = round(importance_sum / stats['total_messages'], 2)
        
        stats['total_sessions'] = len(stats['total_sessions'])
        stats['topics'] = dict(stats['topics'])
        stats['content_types'] = dict(stats['content_types'])
        
        return stats