#!/usr/bin/env python3
"""
Advanced Implementation: BioBERT Integration
Advanced medical NLP using BioBERT transformer models with enhancements
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
warnings.filterwarnings('ignore')

# List of alternative biomedical transformer models for fallback
BIO_MED_MODELS = [
    "dmis-lab/biobert-v1.1",  # BioBERT Base
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # PubMedBERT
    "allenai/scibert_scivocab_uncased",  # SciBERT
    "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"  # BlueBERT
]

try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library available")
except ImportError:
    print("⚠️ Installing transformers library...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True


class BioBERTProcessor:
    """Advanced medical text processing using BioBERT"""
    
    _tokenizer_cache = None
    _model_cache = None
    
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", pooling_strategy="cls", enable_negation_detection=True):
        """
        Initialize BioBERT processor with selectable model and pooling.
        
        pooling_strategy: "cls" or "mean" for last hidden states pooling.
        enable_negation_detection: enable basic negation detection rules.
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        self.enable_negation_detection = enable_negation_detection
        self.negation_keywords = [
            'no history of', 'denies', 'without', 'absence of', 'not currently',
            'never had', 'free of', 'rules out', 'negative for', 'no evidence of'
        ]
        
        # Load model and tokenizer once (cache class variables)
        if not BioBERTProcessor._tokenizer_cache or not BioBERTProcessor._model_cache:
            self._load_model_tokenizer_with_fallback()
        self.tokenizer = BioBERTProcessor._tokenizer_cache
        self.model = BioBERTProcessor._model_cache
        self.model.eval()
        
    def _load_model_tokenizer_with_fallback(self):
        # Try models in BIO_MED_MODELS sequentially until successful load
        # Try explicitly requested model first
        try_models = [self.model_name] + [m for m in BIO_MED_MODELS if m != self.model_name]

        for model_name in try_models:

        
            try:
                print(f"📥 Trying to load model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model.to(self.device)
                BioBERTProcessor._tokenizer_cache = tokenizer
                BioBERTProcessor._model_cache = model
                print(f"✅ Loaded model: {model_name}")
                self.model_name = model_name
                return
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
        # Final fallback to DistilBERT
        print("🔄 Falling back to DistilBERT...")
        fallback_model = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModel.from_pretrained(fallback_model)
        model.to(self.device)
        BioBERTProcessor._tokenizer_cache = tokenizer
        BioBERTProcessor._model_cache = model
        self.model_name = fallback_model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text with selected pooling strategy to get contextual embeddings"""
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            if self.pooling_strategy == "cls":
                sentence_embedding = last_hidden[:, 0, :].cpu().numpy()  # CLS token
            elif self.pooling_strategy == "mean":
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                masked_hidden = last_hidden * mask
                summed = masked_hidden.sum(1)
                counts = mask.sum(1)
                mean_hidden = summed / counts
                sentence_embedding = mean_hidden.cpu().numpy()
            else:
                # default to CLS
                sentence_embedding = last_hidden[:, 0, :].cpu().numpy()
        return sentence_embedding.flatten()

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity with normalized BioBERT embeddings"""
        emb1 = self.encode_text(text1).reshape(1, -1)
        emb2 = self.encode_text(text2).reshape(1, -1)
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = cosine_similarity(emb1_norm, emb2_norm)[0][0]
        return float(similarity)

    def _is_negated(self, context_text: str) -> bool:
        """Simple negation detection using keyword matching"""
        if not self.enable_negation_detection:
            return False
        context_lc = context_text.lower()
        return any(kw in context_lc for kw in self.negation_keywords)

    def extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities with basic keywords and BioBERT contextual validation"""
        sentences = text.split('.')
        medical_entities = []
        # Expanded keywords can be loaded dynamically from ontologies like UMLS or SNOMED (not shown here)
        medical_keywords = [
            'diabetes', 'diabetic', 'hypertension', 'heart failure',
            'myocardial infarction', 'creatinine', 'hba1c', 'glucose',
            'blood pressure', 'ejection fraction', 'metformin', 'insulin'
        ]
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_embedding = self.encode_text(sentence)
            for keyword in medical_keywords:
                if keyword.lower() in sentence.lower():
                    keyword_context = f"The patient has {keyword}"
                    similarity = self.semantic_similarity(sentence, keyword_context)
                    is_negated = self._is_negated(sentence)
                    medical_entities.append({
                        'entity': keyword,
                        'context': sentence,
                        'similarity_score': similarity,
                        'biobert_confidence': similarity * (0 if is_negated else 1),
                        'negated': is_negated,
                        'embedding': sentence_embedding[:10].tolist()
                    })
        medical_entities.sort(key=lambda x: x['biobert_confidence'], reverse=True)
        return medical_entities

    def match_criteria_to_patient(self, criteria_text: str, patient_text: str) -> Dict:
        """Match criteria and patient texts using embeddings and entity-level similarity"""
        criteria_embedding = self.encode_text(criteria_text)
        patient_embedding = self.encode_text(patient_text)
        overall_similarity = cosine_similarity(
            criteria_embedding.reshape(1, -1),
            patient_embedding.reshape(1, -1)
        )[0][0]
        criteria_entities = self.extract_medical_entities(criteria_text)
        patient_entities = self.extract_medical_entities(patient_text)

        entity_matches = []
        for c_entity in criteria_entities:
            best_match = None
            best_score = 0.0
            for p_entity in patient_entities:
                score = self.semantic_similarity(c_entity['context'], p_entity['context'])
                if score > best_score:
                    best_score = score
                    best_match = p_entity
            if best_match and best_score > 0.6:
                entity_matches.append({
                    'criteria_entity': c_entity['entity'],
                    'patient_entity': best_match['entity'],
                    'biobert_similarity': best_score,
                    'criteria_context': c_entity['context'],
                    'patient_context': best_match['context'],
                    'negated': best_match.get('negated', False) or c_entity.get('negated', False)
                })

        return {
            'overall_similarity': float(overall_similarity),
            'entity_matches': entity_matches,
            'criteria_entities_found': len(criteria_entities),
            'patient_entities_found': len(patient_entities),
            'biobert_model': self.model_name
        }


class AdvancedSemanticMatcher:
    """Enhanced semantic matcher with BioBERT ensemble similarity"""

    def __init__(self):
        self.biobert = BioBERTProcessor()
        self.concept_embeddings = {}
        self._precompute_medical_embeddings()

    def _precompute_medical_embeddings(self):
        medical_concepts = [
            "type 2 diabetes mellitus",
            "essential hypertension",
            "chronic heart failure",
            "myocardial infarction",
            "kidney disease",
            "metformin therapy",
            "insulin treatment",
            "ACE inhibitor therapy",
            "beta blocker treatment"
        ]
        print("🧠 Precomputing BioBERT embeddings for medical concepts...")
        for concept in medical_concepts:
            self.concept_embeddings[concept] = self.biobert.encode_text(concept)
        print("✅ Medical concept embeddings ready")

    def enhanced_concept_similarity(self, concept1: str, concept2: str) -> float:
        biobert_sim = self.biobert.semantic_similarity(concept1, concept2)
        try:
            from fuzzywuzzy import fuzz
            fuzzy_sim = fuzz.ratio(concept1.lower(), concept2.lower()) / 100.0
        except ImportError:
            fuzzy_sim = 0.5
        combined_similarity = (0.7 * biobert_sim) + (0.3 * fuzzy_sim)
        return combined_similarity

    def match_patient_to_criteria(self, patient_text: str, criteria_text: str) -> Dict:
        biobert_match = self.biobert.match_criteria_to_patient(criteria_text, patient_text)
        age_info = self._extract_age_biobert(patient_text)
        lab_values = self._extract_lab_values_biobert(patient_text)
        eligibility_score = self._calculate_biobert_eligibility(biobert_match, age_info, lab_values)
        return {
            'biobert_analysis': biobert_match,
            'age_information': age_info,
            'lab_values': lab_values,
            'eligibility_score': eligibility_score,
            'explanation': self._generate_biobert_explanation(biobert_match, eligibility_score)
        }

    def _extract_age_biobert(self, text: str) -> Dict:
        import re
        age_patterns = [
            r'(\d+)[-\s]*(?:year[-\s]*old|yo|y/o|years?\s+old)',
            r'age[:=\s]*(\d+)',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                age_context = f"patient is {age} years old"
                context_embedding = self.biobert.encode_text(age_context)
                return {'age': age, 'context': age_context, 'biobert_embedding': context_embedding[:5].tolist()}
        return {'age': None}

    def _extract_lab_values_biobert(self, text: str) -> List[Dict]:
        import re
        lab_patterns = [
            (r'hba1c\s*(?:is|of|=|:)?\s*(\d+\.?\d*)\s*%?', 'hba1c'),
            (r'creatinine\s*(?:is|of|=|:)?\s*(\d+\.?\d*)', 'creatinine'),
            (r'blood\s*pressure\s*(?:is|of|=|:)?\s*(\d+)/(\d+)', 'blood_pressure'),
        ]
        lab_values = []
        for pattern, lab_name in lab_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context_text = text[max(0, match.start() - 50):match.end() + 50]
                lab_context_similarity = self.biobert.semantic_similarity(context_text, f"patient has {lab_name} level")
                if lab_name == 'blood_pressure':
                    lab_values.extend([
                        {'lab_name': 'systolic_bp', 'value': int(match.group(1)), 'context': context_text,
                         'biobert_confidence': lab_context_similarity},
                        {'lab_name': 'diastolic_bp', 'value': int(match.group(2)), 'context': context_text,
                         'biobert_confidence': lab_context_similarity}
                    ])
                else:
                    lab_values.append({'lab_name': lab_name, 'value': float(match.group(1)), 'context': context_text,
                                       'biobert_confidence': lab_context_similarity})
        return lab_values

    def _calculate_biobert_eligibility(self, biobert_match: Dict, age_info: Dict, lab_values: List[Dict]) -> float:
        base_score = biobert_match['overall_similarity']
        entity_bonus = min(0.2, len(biobert_match['entity_matches']) * 0.05)
        age_factor = 0.1 if age_info.get('age') and 18 <= age_info['age'] <= 75 else 0.0
        lab_factor = min(0.1, len(lab_values) * 0.02)
        final_score = min(1.0, base_score + entity_bonus + age_factor + lab_factor)
        return final_score

    def _generate_biobert_explanation(self, biobert_match: Dict, eligibility_score: float) -> str:
        explanation = f"BioBERT analysis: {eligibility_score:.1%} match confidence. "
        if biobert_match['entity_matches']:
            matched_entities = [match['criteria_entity'] for match in biobert_match['entity_matches'][:3]]
            explanation += f"Key matched concepts: {', '.join(matched_entities)}. "
        if eligibility_score >= 0.8:
            explanation += "Strong semantic alignment with eligibility criteria."
        elif eligibility_score >= 0.6:
            explanation += "Moderate semantic alignment with eligibility criteria."
        else:
            explanation += "Limited semantic alignment with eligibility criteria."
        return explanation


def demo_biobert_integration():
    """Demonstrate BioBERT integration"""
    print("🤖 BioBERT Integration Demo")
    print("=" * 35)

    matcher = AdvancedSemanticMatcher()

    patient_text = """
    Patient is a 52-year-old male with a history of type 2 diabetes mellitus
    diagnosed 5 years ago. Currently taking metformin 1000mg twice daily.
    Recent HbA1c is 8.4%. Blood pressure is well controlled at 128/82 mmHg.
    No history of heart failure or myocardial infarction.
    """

    criteria_text = """
    Inclusion Criteria:
    - Age 18 to 75 years
    - Diagnosed with type 2 diabetes mellitus for at least 6 months
    - HbA1c level between 7.0% and 10.5%
    - Currently taking metformin

    Exclusion Criteria:
    - History of heart failure
    - Recent myocardial infarction
    """

    print("Patient Text:")
    print(patient_text)
    print()

    print("Criteria Text:")
    print(criteria_text)
    print()

    print("🧠 Running BioBERT Analysis...")
    match_result = matcher.match_patient_to_criteria(patient_text, criteria_text)

    print("BioBERT Analysis Results:")
    print("-" * 30)

    biobert_analysis = match_result['biobert_analysis']
    print(f"Overall Semantic Similarity: {biobert_analysis['overall_similarity']:.3f}")
    print(f"Entities in Criteria: {biobert_analysis['criteria_entities_found']}")
    print(f"Entities in Patient: {biobert_analysis['patient_entities_found']}")
    print(f"BioBERT Model: {biobert_analysis['biobert_model']}")

    print("\nEntity Matches:")
    for match in biobert_analysis['entity_matches']:
        print(f"  • '{match['criteria_entity']}' ↔ '{match['patient_entity']}'")
        print(f"    Similarity: {match['biobert_similarity']:.3f}")

    print(f"\nFinal Eligibility Score: {match_result['eligibility_score']:.3f}")
    print(f"Explanation: {match_result['explanation']}")

    print("\n🔍 BioBERT Concept Similarity Examples:")
    concept_pairs = [
        ("diabetes", "diabetic condition"),
        ("type 2 diabetes", "non-insulin dependent diabetes"),
        ("heart failure", "cardiac failure"),
        ("myocardial infarction", "heart attack")
    ]

    for concept1, concept2 in concept_pairs:
        similarity = matcher.enhanced_concept_similarity(concept1, concept2)
        print(f"  '{concept1}' ↔ '{concept2}': {similarity:.3f}")


if __name__ == "__main__":
    demo_biobert_integration()
