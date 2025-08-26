"""
Text preprocessing module for Auslan sign retrieval system.
Handles tokenization, cleaning, and normalization of user input.
"""

import re
from typing import List
from nltk.stem import PorterStemmer

class TextPreprocessor:
    """
    Handles text preprocessing for sign language retrieval.
    Includes tokenization, normalization, and cleaning operations.
    """
    
    def __init__(self):
        """Initialize the preprocessor with common stop words and contractions."""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your'
        }
        
        # Common contractions to expand
        self.contractions = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean input text by removing special characters and normalizing.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation except apostrophes (for remaining contractions)
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()
        
        # Filter out empty tokens and single characters (except 'a', 'i')
        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 or token in ['a', 'i']:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove common stop words from token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens without stop words
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        return [self.stemmer.stem(t) for t in tokens]
    
    def preprocess(self, text: str, remove_stops: bool = False, use_stemming: bool = False) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Raw input text
            remove_stops (bool): Whether to remove stop words
            use_stemming (bool): Whether to apply stemming
            
        Returns:
            List[str]: Processed tokens ready for matching
        """
        tokens = self.tokenize(text)
        
        if remove_stops:
            tokens = self.remove_stop_words(tokens)
        
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def preprocess_batch(self, texts: List[str], remove_stops: bool = False, use_stemming: bool = False) -> List[List[str]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts (List[str]): List of input texts
            remove_stops (bool): Whether to remove stop words
            use_stemming (bool): Whether to apply stemming
            
        Returns:
            List[List[str]]: List of processed token lists
        """
        return [self.preprocess(text, remove_stops, use_stemming) for text in texts]

# Test the preprocessor
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test cases
    test_texts = [
        "Hello, how are you today?",
        "I can't find the toilet.",
        "Let's go buy some food and eat together!",
        "Where is your house? I'd like to visit.",
        "My friend speaks very well."
    ]
    
    print("Testing Text Preprocessor:")
    print("=" * 50)
    
    for text in test_texts:
        tokens = preprocessor.preprocess(text)
        tokens_no_stops = preprocessor.preprocess(text, remove_stops=True)
        
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"No stops: {tokens_no_stops}")
        print("-" * 30)