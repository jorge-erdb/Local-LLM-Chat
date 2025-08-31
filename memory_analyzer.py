import json
import os
import re
import math
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

class MemoryAnalyzer:
    def __init__(self, memory_dir='./memory'):
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, 'memory.jsonl')
        self.index_file = os.path.join(memory_dir, 'index.json')
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Common words to filter out (stop words)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cant',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that',
            'these', 'those', 'what', 'where', 'when', 'why', 'how', 'which', 'who',
            'whom', 'whose', 'if', 'because', 'since', 'while', 'although', 'though',
            'as', 'like', 'than', 'so', 'very', 'just', 'now', 'then', 'here', 'there'
        }
        
        # Topic keywords for classification
        self.topic_keywords = {
            'personal_info': ['name', 'age', 'birthday', 'family', 'friend', 'hobby', 'like', 'favorite', 'love', 'hate'],
            'creative_writing': ['story', 'chapter', 'character', 'plot', 'write', 'writing', 'novel', 'book', 'cyberpunk', 'fiction'],
            'technical': ['code', 'program', 'function', 'class', 'variable', 'python', 'javascript', 'html', 'css', 'api'],
            'conversation': ['hello', 'hi', 'bye', 'goodbye', 'thank', 'thanks', 'sorry', 'please', 'help', 'question'],
            'memory_recall': ['recall', 'remember', 'forgot', 'remind', 'memory', 'before', 'previous', 'last', 'earlier'],
            'problem_solving': ['problem', 'issue', 'error', 'bug', 'fix', 'solution', 'solve', 'help', 'trouble', 'debug'],
            'learning': ['learn', 'teach', 'explain', 'understand', 'know', 'tutorial', 'guide', 'how', 'what', 'why']
        }
        
        # Content type patterns
        self.content_patterns = {
            'question': [r'\?', r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhere\b', r'\bwhen\b', r'\bwho\b', r'\bcan\b', r'\bis\b', r'\bare\b', r'\bdoes\b', r'\bdo\b'],
            'command': [r'^/', r'\bplease\b', r'\bcan you\b', r'\bwould you\b', r'\bcould you\b'],
            'story': [r'\bchapter\b', r'\bonce upon\b', r'\bcharacter\b', r'\bplot\b', r'\.\.\.', r'\bnarrative\b'],
            'code': [r'\bdef\b', r'\bclass\b', r'\bfunction\b', r'```', r'\bimport\b', r'\breturn\b', r'\bif\b.*:', r'\bfor\b.*:'],
            'greeting': [r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgood morning\b', r'\bgood afternoon\b', r'\bgood evening\b'],
            'gratitude': [r'\bthank\b', r'\bthanks\b', r'\bappreciate\b', r'\bgrateful\b'],
            'correction': [r'\bno\b', r'\bnope\b', r'\bactually\b', r'\bwrong\b', r'\bmistake\b', r'\bcorrect\b']
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for search indexing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract meaningful keywords from text"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        # Filter out stop words and very short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get most common words that aren't stop words
        keywords = [word for word, count in word_counts.most_common(max_keywords)]
        
        return keywords

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simple pattern-based approach)"""
        entities = []
        
        # Find capitalized words (potential names)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, text)
        
        # Filter out common words that happen to be capitalized
        common_caps = {'I', 'The', 'A', 'An', 'This', 'That', 'Chapter', 'Hello', 'Hi'}
        for name in potential_names:
            if name not in common_caps and len(name) > 1:
                entities.append(name)
        
        # Find other patterns like dates, numbers, etc.
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        dates = re.findall(date_pattern, text)
        entities.extend(dates)
        
        return list(set(entities))  # Remove duplicates

    def classify_topics(self, text: str, keywords: List[str]) -> List[str]:
        """Classify text into topic categories"""
        text_lower = text.lower()
        topics = []
        
        for topic, topic_words in self.topic_keywords.items():
            # Check if any topic keywords appear in text or extracted keywords
            if any(word in text_lower for word in topic_words) or \
               any(word in keywords for word in topic_words):
                topics.append(topic)
        
        return topics if topics else ['general']

    def determine_content_type(self, text: str) -> str:
        """Determine the type of content"""
        text_lower = text.lower()
        
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return content_type
        
        return 'statement'

    def calculate_importance(self, text: str, role: str, keywords: List[str], entities: List[str]) -> int:
        """Calculate importance score (1-10)"""
        score = 5  # Base score
        
        # Length factor (longer messages might be more important)
        word_count = len(text.split())
        if word_count > 50:
            score += 2
        elif word_count > 20:
            score += 1
        elif word_count < 5:
            score -= 1
        
        # Entity factor (messages with names/dates are often important)
        score += min(len(entities), 2)
        
        # Role factor (user questions often more important for memory)
        if role == 'user':
            score += 1
        
        # Keyword significance
        important_keywords = ['name', 'remember', 'recall', 'important', 'forget', 'story', 'chapter']
        if any(keyword in text.lower() for keyword in important_keywords):
            score += 2
        
        # Question factor
        if '?' in text:
            score += 1
        
        # Command factor
        if text.startswith('/'):
            score += 1
        
        # Ensure score is within bounds
        return max(1, min(10, score))

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 4

    def process_message(self, session_id: str, message_index: int, message: dict, conversation_title: str = None) -> dict:
        """Process a single message and extract metadata"""
        content = message.get('content', '')
        role = message.get('role', '')
        timestamp = message.get('timestamp', '')
        
        # Extract metadata
        normalized_content = self.normalize_text(content)
        keywords = self.extract_keywords(content)
        entities = self.extract_entities(content)
        topics = self.classify_topics(content, keywords)
        content_type = self.determine_content_type(content)
        importance = self.calculate_importance(content, role, keywords, entities)
        token_count = self.count_tokens(content)
        
        # Create memory entry
        memory_entry = {
            'id': f"{session_id}_msg{message_index}",
            'session_id': session_id,
            'message_index': message_index,
            'timestamp': timestamp,
            'role': role,
            'content': content,
            'content_normalized': normalized_content,
            'tokens': token_count,
            'keywords': keywords,
            'entities': entities,
            'topics': topics,
            'importance': importance,
            'content_type': content_type,
            'conversation_title': conversation_title or 'Untitled'
        }
        
        return memory_entry

    def process_conversation_file(self, filepath: str) -> List[dict]:
        """Process an entire conversation file"""
        memory_entries = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = list(f)
            
            if not lines:
                return memory_entries
            
            # Extract session ID from filename
            session_id = os.path.basename(filepath).replace('.jsonl', '')
            
            # Get conversation title from first line
            first_line = json.loads(lines[0].strip())
            conversation_title = first_line.get('title', 'Untitled')
            
            # Process each message (skip title line)
            for i, line in enumerate(lines[1:], 1):
                try:
                    message = json.loads(line.strip())
                    if 'role' in message and 'content' in message:
                        memory_entry = self.process_message(
                            session_id, i, message, conversation_title
                        )
                        memory_entries.append(memory_entry)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            print(f"Error processing conversation file {filepath}: {e}")
        
        return memory_entries

    def rebuild_memory_from_conversations(self, conversations_dir='./conversations'):
        """Rebuild entire memory.jsonl from all conversation files"""
        print("Rebuilding memory database from conversations...")
        
        # Clear existing memory file
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        
        total_processed = 0
        
        if not os.path.exists(conversations_dir):
            print(f"Conversations directory {conversations_dir} not found")
            return
        
        # Process all conversation files
        for filename in os.listdir(conversations_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(conversations_dir, filename)
                print(f"Processing {filename}...")
                
                memory_entries = self.process_conversation_file(filepath)
                
                # Append to memory file
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    for entry in memory_entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                total_processed += len(memory_entries)
        
        print(f"Memory rebuild complete! Processed {total_processed} messages.")
        
        # Build search index
        self.build_search_index()

    def build_search_index(self):
        """Build search index from memory.jsonl"""
        print("Building search index...")
        
        index = {
            'vocabulary': defaultdict(list),      # word -> [message_ids]
            'entities': defaultdict(list),        # entity -> [message_ids]  
            'topics': defaultdict(list),          # topic -> [message_ids]
            'content_types': defaultdict(list),   # type -> [message_ids]
            'sessions': defaultdict(list),        # session -> [message_ids]
            'timestamps': defaultdict(list),      # date -> [message_ids]
            'last_updated': datetime.now().isoformat()
        }
        
        if not os.path.exists(self.memory_file):
            return
        
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    msg_id = entry['id']
                    
                    # Index keywords
                    for keyword in entry.get('keywords', []):
                        index['vocabulary'][keyword].append(msg_id)
                    
                    # Index entities
                    for entity in entry.get('entities', []):
                        index['entities'][entity].append(msg_id)
                    
                    # Index topics
                    for topic in entry.get('topics', []):
                        index['topics'][topic].append(msg_id)
                    
                    # Index content type
                    content_type = entry.get('content_type', 'unknown')
                    index['content_types'][content_type].append(msg_id)
                    
                    # Index session
                    session_id = entry.get('session_id', '')
                    index['sessions'][session_id].append(msg_id)
                    
                    # Index by date (YYYY-MM format for monthly grouping)
                    timestamp = entry.get('timestamp', '')
                    if timestamp:
                        try:
                            date_key = timestamp[:7]  # YYYY-MM
                            index['timestamps'][date_key].append(msg_id)
                        except:
                            pass
                            
                except json.JSONDecodeError:
                    continue
        
        # Convert defaultdicts to regular dicts for JSON serialization
        index = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in index.items()}
        
        # Save index
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        print("Search index built successfully!")

    def add_new_message(self, session_id: str, message_index: int, message: dict, conversation_title: str = None):
        """Add a single new message to memory (for incremental updates)"""
        memory_entry = self.process_message(session_id, message_index, message, conversation_title)
        
        # Append to memory file
        with open(self.memory_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(memory_entry, ensure_ascii=False) + '\n')
        
        # Update search index incrementally
        self._update_index_for_entry(memory_entry)

    def _update_index_for_entry(self, entry: dict):
        """Update search index for a single entry"""
        # Load existing index
        index = {}
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        
        # Initialize index structure if needed
        for key in ['vocabulary', 'entities', 'topics', 'content_types', 'sessions', 'timestamps']:
            if key not in index:
                index[key] = {}
        
        msg_id = entry['id']
        
        # Update vocabulary
        for keyword in entry.get('keywords', []):
            if keyword not in index['vocabulary']:
                index['vocabulary'][keyword] = []
            if msg_id not in index['vocabulary'][keyword]:
                index['vocabulary'][keyword].append(msg_id)
        
        # Update other indices similarly...
        for entity in entry.get('entities', []):
            if entity not in index['entities']:
                index['entities'][entity] = []
            if msg_id not in index['entities'][entity]:
                index['entities'][entity].append(msg_id)
        
        # Save updated index
        index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example usage
    analyzer = MemoryAnalyzer()
    analyzer.rebuild_memory_from_conversations()