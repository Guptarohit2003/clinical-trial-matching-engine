#!/usr/bin/env python3
"""
Clinical Trial Semantic Matching Engine - Day 2 Complete Implementation
Patient Processing & Advanced Semantic Matching System

This builds on Day 1 output and creates complete patient-trial matching
with fuzzy similarity, medical concept mapping, and eligibility scoring.

Author: Your Name
Date: September 2025
Project: Semantic Matching Engine for Clinical Trial Recruitment
"""

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Import Day 1 classes
from advanced_day1_implementation import MedicalConceptExtractor

# Install required packages if not available
try:
    from fuzzywuzzy import fuzz, process
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # print("Installing required packages...")
    # import subprocess
    # import sys

    # subprocess.check_call(
    #     [
    #         sys.executable,
    #         "-m",
    #         "pip",
    #         "install",
    #         "fuzzywuzzy",
    #         "python-levenshtein",
    #         "scikit-learn",
    #     ]
    # )
    # from fuzzywuzzy import fuzz, process
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.metrics.pairwise import cosine_similarity
    print("Error: Required packages are not installed.")
    print(
        "Please ensure you have a requirements.txt file and run 'pip install -r requirements.txt'"
    )
    raise


@dataclass
class MatchingResult:
    """Data class for storing matching results"""

    patient_id: str
    trial_id: str
    overall_score: float
    inclusion_score: float
    exclusion_score: float
    age_eligible: Optional[bool]
    explanation: str
    detailed_matches: Dict


class AdvancedPatientProcessor:
    """Advanced patient clinical text processor with medical NLP"""

    def __init__(self):
        self.concept_extractor = MedicalConceptExtractor()

        # Patient-specific patterns for clinical text
        self.patient_patterns = {
            "demographics": {
                "age": [
                    r"(\d+)[-\s]*(?:year[-\s]*old|yo|y/o|years?\s+old)",
                    r"age[:=\s]*(\d+)",
                    r"(\d+)\s*years?\s*of\s*age",
                ],
                "gender": [
                    r"\b(male|female|man|woman)\b",
                    r"\b(m|f)\s*[/,]",
                    r"gender[:=\s]*(male|female)",
                ],
            },
            "lab_values": {
                "hba1c": [
                    r"hba1c\s*(?:is|of|=|:)?\s*(\d+\.?\d*)\s*%?",
                    r"hemoglobin\s*a1c\s*(?:is|of|=|:)?\s*(\d+\.?\d*)\s*%?",
                    r"a1c\s*(?:level\s*)?(?:is|of|=|:)?\s*(\d+\.?\d*)\s*%?",
                ],
                "ejection_fraction": [
                    r"(?:ejection\s*fraction|ef|lvef)\s*(?:is|of|=|:)?\s*(\d+)\s*%?",
                    r"left\s*ventricular\s*ef\s*(?:is|of|=|:)?\s*(\d+)\s*%?",
                ],
                "blood_pressure": [
                    r"(?:blood\s*pressure|bp)\s*(?:is|of|=|:)?\s*(\d+)/(\d+)",
                    r"systolic\s*(\d+).*?diastolic\s*(\d+)",
                    r"(\d+)/(\d+)\s*mmhg",
                ],
                "bmi": [
                    r"bmi\s*(?:is|of|=|:)?\s*(\d+\.?\d*)",
                    r"body\s*mass\s*index\s*(?:is|of|=|:)?\s*(\d+\.?\d*)",
                ],
                "creatinine": [
                    r"creatinine\s*(?:is|of|=|:)?\s*(\d+\.?\d*)\s*(?:mg/dl)?",
                    r"serum\s*creatinine\s*(?:is|of|=|:)?\s*(\d+\.?\d*)",
                ],
                "egfr": [
                    r"egfr\s*(?:is|of|=|:)?\s*(\d+)\s*(?:ml/min)?",
                    r"estimated\s*gfr\s*(?:is|of|=|:)?\s*(\d+)",
                ],
            },
            "temporal_info": [
                r"diagnosed\s+(?:with\s+)?([^,\.\n]+?)\s+(\d+)\s+(months?|years?)\s+ago",
                r"(?:on|taking)\s+([^,\.\n]+?)\s+for\s+(\d+)\s+(months?|years?)",
                r"history\s+of\s+([^,\.\n]+?)\s+(\d+)\s+(months?|years?)\s+ago",
                r"stable\s+on\s+([^,\.\n]+?)\s+for\s+(\d+)\s+(months?|years?)",
            ],
        }

    def extract_patient_demographics(self, text):
        """Extract patient age and gender"""
        demographics = {}

        # Extract age
        for pattern in self.patient_patterns["demographics"]["age"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 <= age <= 120:  # Reasonable age range
                    demographics["age"] = age
                    break

        # Extract gender
        for pattern in self.patient_patterns["demographics"]["gender"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).lower()
                if gender in ["m", "male", "man"]:
                    demographics["gender"] = "male"
                elif gender in ["f", "female", "woman"]:
                    demographics["gender"] = "female"
                break

        return demographics

    def extract_lab_values(self, text):
        """Extract laboratory values and measurements"""
        lab_values = {}

        for lab_name, patterns in self.patient_patterns["lab_values"].items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        if lab_name == "blood_pressure":
                            # Handle blood pressure specially (systolic/diastolic)
                            systolic = int(match.group(1))
                            diastolic = int(match.group(2))
                            lab_values["systolic_bp"] = systolic
                            lab_values["diastolic_bp"] = diastolic
                        else:
                            value = float(match.group(1))
                            lab_values[lab_name] = value
                        break
                    except (ValueError, IndexError):
                        continue

        return lab_values

    def extract_temporal_information(self, text):
        """Extract temporal information about conditions and treatments"""
        temporal_info = []

        for pattern in self.patient_patterns["temporal_info"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    concept = match.group(1).strip()
                    duration = int(match.group(2))
                    unit = match.group(3).lower().rstrip("s")

                    temporal_info.append(
                        {
                            "concept": concept,
                            "duration": duration,
                            "unit": unit,
                            "original_text": match.group(0),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return temporal_info

    def process_patient_text(self, patient_text, patient_id="unknown"):
        """Complete patient text processing"""
        processed_patient = {
            "patient_id": patient_id,
            "demographics": self.extract_patient_demographics(patient_text),
            "lab_values": self.extract_lab_values(patient_text),
            "medical_concepts": self.concept_extractor.extract_medical_concepts(
                patient_text
            ),
            "temporal_information": self.extract_temporal_information(patient_text),
            "raw_text": patient_text,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "concepts_found": sum(
                    len(concepts)
                    for concepts in self.concept_extractor.extract_medical_concepts(
                        patient_text
                    ).values()
                ),
                "lab_values_found": len(self.extract_lab_values(patient_text)),
                "temporal_info_found": len(
                    self.extract_temporal_information(patient_text)
                ),
            },
        }

        return processed_patient


class AdvancedSemanticMatcher:
    """Advanced semantic matching engine with multiple similarity algorithms"""

    def __init__(self):
        # Advanced medical concept mappings
        self.concept_mappings = {
            # Diabetes variations
            "diabetes": [
                "diabetes",
                "diabetic",
                "dm",
                "t2dm",
                "type 2 diabetes",
                "diabetes mellitus",
            ],
            "diabetic": ["diabetes", "diabetic", "dm", "t2dm", "type 2 diabetes"],
            # Hypertension variations
            "hypertension": [
                "hypertension",
                "htn",
                "high blood pressure",
                "elevated bp",
            ],
            "high_blood_pressure": [
                "hypertension",
                "htn",
                "high blood pressure",
                "elevated bp",
            ],
            # Heart conditions
            "heart_failure": ["heart failure", "hf", "chf", "congestive heart failure"],
            "myocardial_infarction": [
                "myocardial infarction",
                "mi",
                "heart attack",
                "acute mi",
            ],
            # Medications
            "metformin": ["metformin", "glucophage", "metformin hcl"],
            "ace_inhibitor": ["ace inhibitor", "lisinopril", "enalapril", "captopril"],
            "beta_blocker": ["beta blocker", "atenolol", "metoprolol", "propranolol"],
        }

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", max_features=1000, ngram_range=(1, 2)
        )

        # Similarity weights
        self.similarity_weights = {
            "exact_match": 1.0,
            "concept_mapping": 0.9,
            "fuzzy_high": 0.8,
            "fuzzy_medium": 0.6,
            "fuzzy_low": 0.3,
            "tfidf_high": 0.7,
            "tfidf_medium": 0.5,
        }

    def compute_concept_similarity(self, concept1, concept2):
        """Compute semantic similarity between medical concepts"""
        concept1_lower = concept1.lower().replace("_", " ")
        concept2_lower = concept2.lower().replace("_", " ")

        # Exact match
        if concept1_lower == concept2_lower:
            return self.similarity_weights["exact_match"]

        # Check concept mappings
        max_mapping_score = 0.0
        for main_concept, variations in self.concept_mappings.items():
            if concept1_lower in variations and concept2_lower in variations:
                max_mapping_score = max(
                    max_mapping_score, self.similarity_weights["concept_mapping"]
                )
            elif concept1_lower == main_concept and concept2_lower in variations:
                max_mapping_score = max(
                    max_mapping_score, self.similarity_weights["concept_mapping"]
                )
            elif concept2_lower == main_concept and concept1_lower in variations:
                max_mapping_score = max(
                    max_mapping_score, self.similarity_weights["concept_mapping"]
                )

        if max_mapping_score > 0:
            return max_mapping_score

        # Fuzzy string matching
        fuzzy_ratio = fuzz.ratio(concept1_lower, concept2_lower) / 100.0
        fuzzy_partial = fuzz.partial_ratio(concept1_lower, concept2_lower) / 100.0
        fuzzy_token_set = fuzz.token_set_ratio(concept1_lower, concept2_lower) / 100.0

        max_fuzzy = max(fuzzy_ratio, fuzzy_partial, fuzzy_token_set)

        if max_fuzzy >= 0.9:
            return self.similarity_weights["fuzzy_high"] * max_fuzzy
        elif max_fuzzy >= 0.7:
            return self.similarity_weights["fuzzy_medium"] * max_fuzzy
        elif max_fuzzy >= 0.5:
            return self.similarity_weights["fuzzy_low"] * max_fuzzy

        # TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                [concept1_lower, concept2_lower]
            )
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[
                0
            ][0]

            if tfidf_similarity >= 0.7:
                return self.similarity_weights["tfidf_high"] * tfidf_similarity
            elif tfidf_similarity >= 0.4:
                return self.similarity_weights["tfidf_medium"] * tfidf_similarity
        except:
            pass

        return 0.0

    def check_age_eligibility(self, trial_age_criteria, patient_age):
        """Check age eligibility with detailed scoring"""
        if not trial_age_criteria or patient_age is None:
            return {
                "eligible": None,
                "score": 0.5,
                "confidence": 0.0,
                "explanation": "Age information missing",
            }

        # Handle multiple age criteria
        best_match = {
            "eligible": False,
            "score": 0.0,
            "confidence": 0.0,
            "explanation": "",
        }

        for age_criterion in trial_age_criteria:
            min_age = age_criterion["min_age"]
            max_age = age_criterion["max_age"]
            score = 0.0

            if min_age <= patient_age <= max_age:
                # Perfect match
                score = 1.0
                confidence = 1.0
                explanation = (
                    f"Age {patient_age} within required range {min_age}-{max_age} years"
                )
                return {
                    "eligible": True,
                    "score": score,
                    "confidence": confidence,
                    "explanation": explanation,
                }
            else:
                # Calculate proximity score for near misses
                if patient_age < min_age:
                    distance = min_age - patient_age
                    if distance <= 5:  # Within 5 years
                        score = max(0.1, 1.0 - (distance / 10))
                        explanation = f"Age {patient_age} is {distance} years below minimum age {min_age}"
                elif patient_age > max_age:
                    distance = patient_age - max_age
                    if distance <= 5:  # Within 5 years
                        score = max(0.1, 1.0 - (distance / 10))
                        explanation = f"Age {patient_age} is {distance} years above maximum age {max_age}"

                if score > best_match["score"]:
                    best_match = {
                        "eligible": False,
                        "score": score,
                        "confidence": 0.8,
                        "explanation": explanation,
                    }

        return (
            best_match
            if best_match["score"] > 0
            else {
                "eligible": False,
                "score": 0.0,
                "confidence": 1.0,
                "explanation": f"Age {patient_age} outside all acceptable ranges",
            }
        )

    def check_numeric_eligibility(self, trial_numeric_criteria, patient_lab_values):
        """Check numeric/lab value eligibility"""
        numeric_matches = []

        for criterion in trial_numeric_criteria:
            parameter = criterion["parameter"].lower()

            # Find matching patient lab value
            patient_value = None
            matched_param = None
            score = 0.0

            # Direct matching
            if parameter in patient_lab_values:
                patient_value = patient_lab_values[parameter]
                matched_param = parameter
            else:
                # Fuzzy matching of parameter names
                for patient_param, value in patient_lab_values.items():
                    similarity = self.compute_concept_similarity(
                        parameter, patient_param
                    )
                    if similarity > 0.7:
                        patient_value = value
                        matched_param = patient_param
                        break

            if patient_value is None:
                numeric_matches.append(
                    {
                        "parameter": parameter,
                        "eligible": None,
                        "score": 0.2,
                        "confidence": 0.0,
                        "explanation": f"No {parameter} value found in patient data",
                    }
                )
                continue

            # Check eligibility based on criterion type
            if criterion["type"] == "range":
                min_val = criterion["min_value"]
                max_val = criterion["max_value"]
                eligible = min_val <= patient_value <= max_val

                if eligible:
                    score = 1.0
                    explanation = f'{parameter}: {patient_value} within range {min_val}-{max_val} {criterion["unit"]}'
                else:
                    # Calculate proximity score
                    if patient_value < min_val:
                        distance = abs(patient_value - min_val) / min_val
                    else:
                        distance = abs(patient_value - max_val) / max_val

                    score = max(0.0, 1.0 - distance) if distance < 0.5 else 0.0
                    explanation = f'{parameter}: {patient_value} outside range {min_val}-{max_val} {criterion["unit"]}'

            elif criterion["type"] == "comparison":
                operator = criterion["operator"]
                threshold = criterion["value"]

                if operator in [">", ">=", "≥"]:
                    eligible = patient_value >= threshold
                elif operator in ["<", "<=", "≤"]:
                    eligible = patient_value <= threshold
                elif operator in ["=", "=="]:
                    eligible = abs(patient_value - threshold) / threshold < 0.1
                else:
                    eligible = False

                if eligible:
                    score = 1.0
                else:
                    # Proximity scoring for near misses
                    distance = abs(patient_value - threshold) / threshold
                    score = max(0.0, 1.0 - distance) if distance < 0.3 else 0.0

                explanation = f'{parameter}: {patient_value} {operator} {threshold} {criterion["unit"]} = {eligible}'

            numeric_matches.append(
                {
                    "parameter": parameter,
                    "patient_parameter": matched_param,
                    "patient_value": patient_value,
                    "eligible": eligible,
                    "score": score,
                    "confidence": 0.9,
                    "explanation": explanation,
                }
            )

        return numeric_matches

    def match_medical_concepts(self, trial_concepts, patient_concepts):
        """Match medical concepts between trial criteria and patient data"""
        concept_matches = []

        for category in ["conditions", "medications", "measurements"]:
            trial_category_concepts = trial_concepts.get(category, [])
            patient_category_concepts = patient_concepts.get(category, [])

            for trial_concept in trial_category_concepts:
                trial_term = trial_concept["concept"]
                trial_negated = trial_concept.get("is_negated", False)

                best_match = None
                best_similarity = 0.0

                for patient_concept in patient_category_concepts:
                    patient_term = patient_concept["concept"]
                    similarity = self.compute_concept_similarity(
                        trial_term, patient_term
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = patient_concept

                # Calculate final score based on similarity and negation logic
                if best_match and best_similarity > 0.5:
                    patient_negated = best_match.get("is_negated", False)

                    if trial_negated and patient_negated:
                        # Both negated - good match (criterion requires absence, patient doesn't have it)
                        final_score = best_similarity
                        match_type = "Both absent - good match"
                    elif not trial_negated and not patient_negated:
                        # Both positive - good match (criterion requires presence, patient has it)
                        final_score = best_similarity
                        match_type = "Both present - good match"
                    elif trial_negated and not patient_negated:
                        # Trial excludes, patient has - poor match
                        final_score = 0.1
                        match_type = "Exclusion violated - poor match"
                    else:
                        # Trial requires, patient doesn't have - poor match
                        final_score = 0.1
                        match_type = "Requirement not met - poor match"
                else:
                    final_score = 0.0
                    match_type = "No matching concept found"
                    best_match = None

                concept_matches.append(
                    {
                        "trial_concept": trial_term,
                        "patient_match": best_match["concept"] if best_match else None,
                        "category": category,
                        "similarity_score": best_similarity,
                        "final_score": final_score,
                        "trial_negated": trial_negated,
                        "patient_negated": (
                            best_match.get("is_negated", False) if best_match else False
                        ),
                        "match_type": match_type,
                        "explanation": f"{trial_term} ({'excluded' if trial_negated else 'required'}): {match_type}",
                    }
                )

        return concept_matches

    def compute_overall_scores(self, age_match, numeric_matches, concept_matches):
        """Compute comprehensive eligibility scores"""
        scores = []
        explanations = []

        # Age scoring (high weight)
        if age_match["score"] is not None:
            scores.extend([age_match["score"]] * 3)  # Triple weight for age
            if age_match["eligible"] is not None:
                explanations.append(age_match["explanation"])

        # Numeric criteria scoring
        for match in numeric_matches:
            if match["score"] is not None:
                scores.append(match["score"])
                if match["eligible"] is not None:
                    explanations.append(match["explanation"])

        # Concept matching scoring
        for match in concept_matches:
            scores.append(match["final_score"])
            if match["final_score"] > 0.7:
                explanations.append(match["explanation"])

        overall_score = np.mean(scores) if scores else 0.0

        return overall_score, explanations[:5]  # Top 5 explanations

    def match_patient_to_trial(self, processed_trial, processed_patient):
        """Complete patient-to-trial matching with detailed analysis"""
        inclusion_criteria = processed_trial["inclusion_criteria"]
        exclusion_criteria = processed_trial["exclusion_criteria"]

        # Inclusion criteria matching
        inclusion_age_match = self.check_age_eligibility(
            inclusion_criteria["age_criteria"],
            processed_patient["demographics"].get("age"),
        )

        inclusion_numeric_matches = self.check_numeric_eligibility(
            inclusion_criteria["numeric_criteria"], processed_patient["lab_values"]
        )

        inclusion_concept_matches = self.match_medical_concepts(
            inclusion_criteria["medical_concepts"],
            processed_patient["medical_concepts"],
        )

        # Exclusion criteria matching
        exclusion_age_match = self.check_age_eligibility(
            exclusion_criteria["age_criteria"],
            processed_patient["demographics"].get("age"),
        )

        exclusion_numeric_matches = self.check_numeric_eligibility(
            exclusion_criteria["numeric_criteria"], processed_patient["lab_values"]
        )

        exclusion_concept_matches = self.match_medical_concepts(
            exclusion_criteria["medical_concepts"],
            processed_patient["medical_concepts"],
        )

        # Compute scores
        inclusion_score, inclusion_explanations = self.compute_overall_scores(
            inclusion_age_match, inclusion_numeric_matches, inclusion_concept_matches
        )

        exclusion_score, exclusion_explanations = self.compute_overall_scores(
            exclusion_age_match, exclusion_numeric_matches, exclusion_concept_matches
        )

        # Overall eligibility (high inclusion, low exclusion)
        # Exclusion criteria violations should significantly reduce eligibility
        overall_score = inclusion_score * (1.0 - exclusion_score * 0.7)

        # Generate comprehensive explanation
        explanation_parts = []
        if inclusion_explanations:
            explanation_parts.append(
                f"Inclusion: {'; '.join(inclusion_explanations[:2])}"
            )
        if exclusion_explanations:
            explanation_parts.append(
                f"Exclusion: {'; '.join(exclusion_explanations[:2])}"
            )

        final_explanation = (
            "; ".join(explanation_parts)
            if explanation_parts
            else "Limited matching criteria found"
        )

        return MatchingResult(
            patient_id=processed_patient["patient_id"],
            trial_id=processed_trial["trial_id"],
            overall_score=overall_score,
            inclusion_score=inclusion_score,
            exclusion_score=exclusion_score,
            age_eligible=inclusion_age_match["eligible"],
            explanation=final_explanation,
            detailed_matches={
                "inclusion_age": inclusion_age_match,
                "inclusion_numeric": inclusion_numeric_matches,
                "inclusion_concepts": inclusion_concept_matches,
                "exclusion_age": exclusion_age_match,
                "exclusion_numeric": exclusion_numeric_matches,
                "exclusion_concepts": exclusion_concept_matches,
            },
        )


class ComprehensivePatientDatabase:
    """Comprehensive patient database with diverse medical scenarios"""

    def __init__(self):
        self.patients = [
            {
                "patient_id": "PT001",
                "clinical_text": """
                Patient is a 52-year-old male with a 8-year history of type 2 diabetes mellitus. 
                Currently taking metformin 1000mg twice daily for the past 18 months. 
                Recent laboratory results show HbA1c of 8.4%, fasting glucose 165 mg/dL. 
                BMI is 31.2 kg/m². Blood pressure well controlled at 128/82 mmHg on lisinopril. 
                Patient denies chest pain, shortness of breath, or lower extremity edema. 
                No history of heart failure, stroke, or myocardial infarction. 
                Creatinine 1.1 mg/dL, eGFR 88 mL/min/1.73m². 
                Patient is motivated and compliant with medications.
                """,
            },
            {
                "patient_id": "PT002",
                "clinical_text": """
                58-year-old female with essential hypertension diagnosed 6 years ago. 
                Current blood pressure 148/94 mmHg despite being on amlodipine 10mg daily 
                and hydrochlorothiazide 25mg daily for 8 months. Previous ACE inhibitor 
                was discontinued due to dry cough. No diabetes mellitus. 
                BMI 28.5 kg/m². Recent echocardiogram shows normal left ventricular 
                function with ejection fraction 62%. No history of myocardial infarction 
                or stroke. Creatinine 0.9 mg/dL, eGFR 72 mL/min/1.73m². 
                Cholesterol well controlled on atorvastatin.
                """,
            },
            {
                "patient_id": "PT003",
                "clinical_text": """
                67-year-old male with chronic heart failure, NYHA Class II symptoms. 
                History of anterior wall myocardial infarction 14 months ago with 
                subsequent development of heart failure. Current ejection fraction 38% 
                on recent echocardiogram. On optimal medical therapy including 
                lisinopril 10mg daily, metoprolol 50mg twice daily, and furosemide 40mg daily 
                for the past 10 months. No recent hospitalizations in past 6 weeks. 
                Creatinine 1.3 mg/dL, eGFR 58 mL/min/1.73m². No diabetes mellitus. 
                Able to walk one block without significant dyspnea.
                """,
            },
            {
                "patient_id": "PT004",
                "clinical_text": """
                45-year-old female with poorly controlled type 2 diabetes and hypertension. 
                Diagnosed with diabetes 12 years ago, currently on metformin and insulin glargine. 
                HbA1c 9.8%, blood pressure 156/98 mmHg. History of myocardial infarction 
                8 months ago. BMI 34.8 kg/m². Taking lisinopril, atorvastatin, and aspirin. 
                Recent stress test shows no evidence of active ischemia. 
                Creatinine 1.0 mg/dL, eGFR 78 mL/min/1.73m². 
                Patient struggles with medication adherence and dietary compliance.
                """,
            },
            {
                "patient_id": "PT005",
                "clinical_text": """
                73-year-old male with atrial fibrillation and chronic kidney disease stage 3B. 
                Persistent atrial fibrillation diagnosed 3 years ago, currently on warfarin 
                with INR goal 2.0-3.0. Recent INR 2.4. CHA2DS2-VASc score 4 (age, hypertension, diabetes, prior stroke). 
                eGFR 42 mL/min/1.73m², creatinine 1.8 mg/dL, stable for past 6 months. 
                History of ischemic stroke 2.5 years ago with good recovery. 
                Blood pressure controlled on amlodipine. HbA1c 7.2% on metformin. 
                No recent bleeding episodes or hospitalizations.
                """,
            },
            {
                "patient_id": "PT006",
                "clinical_text": """
                39-year-old male post-acute myocardial infarction 45 days ago. 
                Underwent primary PCI with stent placement to LAD. Current medications include 
                dual antiplatelet therapy (aspirin and clopidogrel), atorvastatin 80mg daily, 
                metoprolol 100mg twice daily. Recent LDL cholesterol 95 mg/dL on high-dose statin. 
                Echo shows ejection fraction 52%, mild hypokinesis of anterior wall. 
                No diabetes mellitus. Creatinine 0.8 mg/dL, eGFR >90 mL/min/1.73m². 
                Cardiac rehabilitation initiated. Patient motivated for aggressive risk factor modification.
                """,
            },
        ]

    def get_patients_dataframe(self):
        """Return patients as DataFrame"""
        return pd.DataFrame(self.patients)


def main():
    """Main execution function for Day 2"""
    print("🏥 Clinical Trial Semantic Matching Engine - Day 2 Implementation")
    print("=" * 70)
    print("Patient Processing & Advanced Semantic Matching")

    # Load processed trials from Day 1
    try:
        with open("processed_trials_advanced_day1.json", "r") as f:
            day1_data = json.load(f)
        processed_trials = day1_data["processed_trials"]
        print(f"\n✅ Loaded {len(processed_trials)} processed trials from Day 1")
    except FileNotFoundError:
        print("\n❌ Error: processed_trials_advanced_day1.json not found.")
        print("Please run Day 1 (advanced_day1_implementation.py) first!")
        return

    # Initialize components
    patient_processor = AdvancedPatientProcessor()
    semantic_matcher = AdvancedSemanticMatcher()
    patient_db = ComprehensivePatientDatabase()

    # Process patients
    patients_df = patient_db.get_patients_dataframe()
    processed_patients = []

    print(f"\n👥 Processing {len(patients_df)} diverse patient cases:")

    for _, patient in patients_df.iterrows():
        print(f"\n🔍 Processing {patient['patient_id']}...")

        processed_patient = patient_processor.process_patient_text(
            patient["clinical_text"], patient["patient_id"]
        )
        processed_patients.append(processed_patient)

        # Display processing results
        metadata = processed_patient["processing_metadata"]
        demographics = processed_patient["demographics"]
        lab_values = processed_patient["lab_values"]

        print(
            f"  📊 Age: {demographics.get('age', 'Unknown')}, Gender: {demographics.get('gender', 'Unknown')}"
        )
        print(
            f"  🔬 Lab values: {len(lab_values)} found ({', '.join(lab_values.keys())})"
        )
        print(f"  🏥 Medical concepts: {metadata['concepts_found']} total")
        print(f"  ⏰ Temporal info: {metadata['temporal_info_found']} items")

    # Perform comprehensive matching
    print(f"\n🎯 Performing comprehensive patient-trial matching...")
    all_matching_results = []

    total_combinations = len(processed_patients) * len(processed_trials)
    combination_count = 0

    for processed_patient in processed_patients:
        patient_matches = []
        print(f"\n👤 Matching {processed_patient['patient_id']}:")

        for processed_trial in processed_trials:
            combination_count += 1
            match_result = semantic_matcher.match_patient_to_trial(
                processed_trial, processed_patient
            )
            patient_matches.append(match_result)

        # Sort matches by overall score
        patient_matches.sort(key=lambda x: x.overall_score, reverse=True)

        # Display top 3 matches
        print("  🏆 Top 3 matches:")
        for i, match in enumerate(patient_matches[:3]):
            print(
                f"    {i+1}. {processed_trials[next(j for j, t in enumerate(processed_trials) if t['trial_id'] == match.trial_id)]['condition'][:30]}... "
                f"(Score: {match.overall_score:.3f})"
            )
            print(f"       {match.explanation[:80]}...")

        all_matching_results.append(
            {
                "patient_id": processed_patient["patient_id"],
                "patient_data": processed_patient,
                "matches": [
                    match.__dict__ for match in patient_matches
                ],  # Convert to dict for JSON serialization
            }
        )

    # Calculate comprehensive statistics
    all_scores = [
        match["overall_score"]
        for patient_result in all_matching_results
        for match in patient_result["matches"]
    ]
    high_quality_matches = sum(1 for score in all_scores if score >= 0.7)
    good_matches = sum(1 for score in all_scores if 0.5 <= score < 0.7)

    statistics = {
        "total_patient_trial_combinations": total_combinations,
        "average_matching_score": np.mean(all_scores),
        "score_statistics": {
            "min_score": min(all_scores),
            "max_score": max(all_scores),
            "std_score": np.std(all_scores),
            "median_score": np.median(all_scores),
        },
        "match_quality_distribution": {
            "excellent_matches_0.8_plus": sum(1 for s in all_scores if s >= 0.8),
            "high_quality_matches_0.7_plus": high_quality_matches,
            "good_matches_0.5_to_0.7": good_matches,
            "poor_matches_below_0.5": sum(1 for s in all_scores if s < 0.5),
        },
        "processing_performance": {
            "patients_processed": len(processed_patients),
            "trials_processed": len(processed_trials),
            "total_concepts_extracted": sum(
                p["processing_metadata"]["concepts_found"] for p in processed_patients
            ),
            "total_lab_values_extracted": sum(
                p["processing_metadata"]["lab_values_found"] for p in processed_patients
            ),
        },
    }

    # Save comprehensive results
    output_data = {
        "matching_results": all_matching_results,
        "processed_patients_summary": [
            {
                "patient_id": p["patient_id"],
                "age": p["demographics"].get("age"),
                "gender": p["demographics"].get("gender"),
                "lab_values": list(p["lab_values"].keys()),
                "concepts_found": p["processing_metadata"]["concepts_found"],
            }
            for p in processed_patients
        ],
        "statistics": statistics,
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "system_version": "Clinical Trial Matching Engine v2.0",
            "processing_time": "Real-time",
        },
    }

    filename = "comprehensive_matching_results_day2.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Final comprehensive summary
    print(f"\n📊 Comprehensive Matching Analysis:")
    print(
        f"  🎯 Total combinations evaluated: {statistics['total_patient_trial_combinations']}"
    )
    print(f"  📈 Average matching score: {statistics['average_matching_score']:.3f}")
    print(
        f"  🏆 Excellent matches (≥0.8): {statistics['match_quality_distribution']['excellent_matches_0.8_plus']}"
    )
    print(
        f"  ✅ High quality matches (≥0.7): {statistics['match_quality_distribution']['high_quality_matches_0.7_plus']}"
    )
    print(
        f"  👍 Good matches (0.5-0.7): {statistics['match_quality_distribution']['good_matches_0.5_to_0.7']}"
    )
    print(
        f"  👎 Poor matches (<0.5): {statistics['match_quality_distribution']['poor_matches_below_0.5']}"
    )

    print(f"\n🔬 Processing Performance:")
    print(
        f"  👥 Patients processed: {statistics['processing_performance']['patients_processed']}"
    )
    print(
        f"  🧪 Clinical trials processed: {statistics['processing_performance']['trials_processed']}"
    )
    print(
        f"  🏥 Medical concepts extracted: {statistics['processing_performance']['total_concepts_extracted']}"
    )
    print(
        f"  🔬 Lab values extracted: {statistics['processing_performance']['total_lab_values_extracted']}"
    )

    print(f"\n💾 Results saved to: {filename}")
    print("✅ Day 2 Complete! Advanced semantic matching system ready.")
    print("\n🚀 Next: Run Day 3 for complete API and web interface")

    return all_matching_results, statistics


if __name__ == "__main__":
    try:
        results, stats = main()

        # Quick validation
        print(f"\n🧪 System Validation:")
        best_matches = []
        for patient_result in results:
            best_match = max(
                patient_result["matches"], key=lambda x: x["overall_score"]
            )
            best_matches.append(best_match)
            print(
                f"  {patient_result['patient_id']}: Best match score {best_match['overall_score']:.3f}"
            )

        print(
            f"\n🎯 Overall System Performance: {np.mean([m['overall_score'] for m in best_matches]):.3f}"
        )
        print("✅ Advanced semantic matching system validated and ready!")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\nPlease ensure:")
        print(
            "1. Day 1 has been completed (processed_trials_advanced_day1.json exists)"
        )
        print(
            "2. Required packages are installed: pip install fuzzywuzzy python-levenshtein scikit-learn"
        )
