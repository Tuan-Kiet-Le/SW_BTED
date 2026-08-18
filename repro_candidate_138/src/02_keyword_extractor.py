import nltk
from rake_nltk import Rake
import re

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    if not text:
        return ""
    # Remove HTML-like tags, list indicators, bullet points
    # (Exclude 'o' from character class, handle standalone 'o' list indicators separately)
    text = re.sub(r'[●\t•*\-]', ' ', text)
    text = re.sub(r'(?:^|(?<=\s))o\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords_from_text(text, top_k=8):
    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < 5:
        return []
    
    # Initialize Rake with standard English stopwords
    r = Rake()
    r.extract_keywords_from_text(cleaned)
    # Get keywords with scores
    ranked_phrases = r.get_ranked_phrases_with_scores()
    
    # Filter out very short phrases or numbers
    valid_phrases = []
    for score, phrase in ranked_phrases:
        phrase_clean = phrase.strip().lower()
        # Ensure it contains letters and has length > 2
        if len(phrase_clean) > 2 and re.search(r'[a-zA-Z]', phrase_clean):
            valid_phrases.append((phrase_clean, round(score, 2)))
            if len(valid_phrases) >= top_k:
                break
                
    return valid_phrases

def run_extraction_on_sections(sections_dict, top_k=8):
    keywords_dict = {}
    for sec_id, text in sections_dict.items():
        if text:
            keywords_dict[sec_id] = extract_keywords_from_text(text, top_k=top_k)
        else:
            keywords_dict[sec_id] = []
    return keywords_dict
