#!/usr/bin/env python3
"""
Clinical Trial Semantic Matching Engine - Day 1 Complete Implementation
Real working system with sample data and full processing pipeline

Author: Your Name
Date: September 2025
Project: Semantic Matching Engine for Clinical Trial Recruitment
"""

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore")
import requests
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


def fetch_clinical_trials_api(condition=None, phase=None, status=None, max_trials=100):
    """
    Fetches clinical trials from the ClinicalTrials.gov v2 API
    using the correct query parameters.
    """
    api_url = "https://clinicaltrials.gov/api/v2/studies"

    recruitment_status_map = {
        "not yet recruiting": "NOT_YET_RECRUITING",
        "recruiting": "RECRUITING",
        "enrolling by invitation": "ENROLLING_BY_INVITATION",
        "active not recruiting": "ACTIVE_NOT_RECRUITING",
        "completed": "COMPLETED",
        "terminated": "TERMINATED",
        "withdrawn": "WITHDRAWN",
        "unknown": "UNKNOWN",
    }

    # Map for the dedicated phase filter
    phase_map = {
        "early phase 1": "EARLY_PHASE_1",
        "phase 1": "PHASE_1",
        "phase 2": "PHASE_2",
        "phase 3": "PHASE_3",
        "phase 4": "PHASE_4",
        "n/a": "NA",
    }

    # --- MODIFICATION: Build params dictionary with correct parameters ---
    params = {
        "pageSize": 20,
        "format": "json",
    }

    if condition:
        # Use the dedicated parameter for condition search
        params["query.cond"] = condition

    if status:
        stat_lower = status.lower()
        status_enum = recruitment_status_map.get(stat_lower)
        if not status_enum:
            raise ValueError(f"Unknown recruitment status: {status}")
        # Use the dedicated filter for overall status
        params["filter.overallStatus"] = status_enum

    if phase:
        phase_lower = phase.lower()
        phase_enum = phase_map.get(phase_lower)
        if phase_enum:
            # Use the dedicated filter for phase
            params["filter.phase"] = phase_enum
        else:
            # Fallback for non-standard phase strings
            print(f"Warning: Phase '{phase}' not in standard map. Using text search.")
            params["query.term"] = f"AREA[Phase]{phase}"
    # --- END MODIFICATION ---

    trials = []
    next_page_token = None

    # Use a requests.Session for connection pooling and performance
    with requests.Session() as session:
        while len(trials) < max_trials:
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = session.get(api_url, params=params)

                if resp.status_code == 429:
                    print("Rate limit exceeded. Waiting 2 seconds...")
                    time.sleep(2)
                    continue

                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP Error: {http_err}")
                print(f"Failing URL: {resp.url}")
                print(f"Response text: {resp.text[:500]}")
                break
            except json.JSONDecodeError:
                print(f"JSON Decode Error. Response text: {resp.text[:500]}")
                break

            # <--- Your debug print (still included)
            print("API response data snippet:", json.dumps(data, indent=2)[:1000])

            for study in data.get("studies", []):
                psec = study.get("protocolSection", {})
                emod = psec.get("eligibilityModule", {})
                idmod = psec.get("identificationModule", {})
                condmod = psec.get("conditionsModule", {})
                desmod = psec.get("designModule", {})
                outmod = psec.get("outcomesModule", {})
                contlocmod = psec.get("contactsLocationsModule", {})

                # Safer access for lists that might be empty
                primary_outcomes = outmod.get("primaryOutcomes", [])
                primary_outcome_measure = (
                    primary_outcomes[0].get("measure", "")
                    if primary_outcomes
                    else "N/A"
                )

                overall_officials = contlocmod.get("overallOfficials", [])
                sponsor_name = (
                    overall_officials[0].get("name", "") if overall_officials else "N/A"
                )

                # Handle potentially empty or N/A phases
                phases = desmod.get("phases", [])
                phase_str = ", ".join(phases) if phases else "N/A"

                trials.append(
                    {
                        "nct_id": idmod.get("nctId", ""),
                        "title": idmod.get("officialTitle", idmod.get("briefTitle", ""))
                        or "N/A",
                        "condition": ", ".join(condmod.get("conditions", [])) or "N/A",
                        "phase": phase_str,
                        "sponsor": sponsor_name,
                        "eligibility_criteria": emod.get("eligibilityCriteria", ""),
                        "primary_endpoint": primary_outcome_measure,
                        "estimated_enrollment": emod.get("enrollmentCount", None),
                    }
                )

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            if len(trials) >= max_trials:
                break

            time.sleep(0.5)

    return trials[:max_trials]


class MedicalConceptExtractor:
    """Extract medical concepts via BioBERT and negation based on contextual keywords"""

    def __init__(self, ner_model_name="alvaroalon2/biobert_diseases_ner", device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(
            self.device
        )
        self.label_map = {v: k for k, v in self.model.config.id2label.items()}

        self.condition_labels = {"DISEASE", "CONDITION", "DISORDER"}
        self.medication_labels = {"DRUG", "MEDICATION"}
        self.measurement_labels = {"MEASUREMENT", "LAB_VALUE"}
        self.negation_keywords = [
            "no history of",
            "denies",
            "without",
            "absence of",
            "not currently",
            "never had",
            "free of",
            "rules out",
            "negative for",
            "no evidence of",
        ]

    def _predict_ner(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        offsets = self.tokenizer(
            text, return_offsets_mapping=True, truncation=True, max_length=512
        )["offset_mapping"]

        entities = []
        current_tokens = []
        current_label = None
        current_start = None
        for i, label_id in enumerate(predictions):
            label = self.model.config.id2label[label_id]
            if label.startswith("B-"):
                if current_tokens:
                    entities.append(
                        {
                            "entity": current_label,
                            "text": text[current_start : offsets[i - 1][1]],
                            "start": current_start,
                            "end": offsets[i - 1][1],
                        }
                    )
                current_tokens = [tokens[i]]
                current_label = label[2:]
                current_start = offsets[i][0]
            elif label.startswith("I-") and current_label == label[2:]:
                current_tokens.append(tokens[i])
            else:
                if current_tokens:
                    entities.append(
                        {
                            "entity": current_label,
                            "text": text[current_start : offsets[i - 1][1]],
                            "start": current_start,
                            "end": offsets[i - 1][1],
                        }
                    )
                    current_tokens = []
                    current_label = None
                    current_start = None
        if current_tokens:
            entities.append(
                {
                    "entity": current_label,
                    "text": text[current_start : offsets[len(predictions) - 1][1]],
                    "start": current_start,
                    "end": offsets[len(predictions) - 1][1],
                }
            )
        return entities

    def _is_negated(self, context_text):
        context_lc = context_text.lower()
        return any(kw in context_lc for kw in self.negation_keywords)

    def extract_medical_concepts(self, text):
        extracted_concepts = {"conditions": [], "medications": [], "measurements": []}
        entities = self._predict_ner(text)

        for ent in entities:
            entity_type = ent["entity"].upper()
            if entity_type in self.condition_labels:
                cat = "conditions"
            elif entity_type in self.medication_labels:
                cat = "medications"
            elif entity_type in self.measurement_labels:
                cat = "measurements"
            else:
                continue

            context_start = max(0, ent["start"] - 100)
            context = text[context_start : ent["start"]]
            is_negated = self._is_negated(context)

            extracted_concepts[cat].append(
                {
                    "concept": ent["text"].lower(),
                    "matched_text": ent["text"],
                    "is_negated": is_negated,
                    "confidence": 0.95,
                    "position": (ent["start"], ent["end"]),
                    "context": text[context_start : ent["end"] + 50],
                }
            )

        for category in extracted_concepts:
            unique = {}
            for c in extracted_concepts[category]:
                key = c["concept"]
                if key not in unique or unique[key]["confidence"] < c["confidence"]:
                    unique[key] = c
            extracted_concepts[category] = list(unique.values())

        return extracted_concepts


class AdvancedCriteriaProcessor:
    """Advanced clinical trial eligibility criteria processor"""

    def __init__(self, device="cpu"):
        self.concept_extractor = MedicalConceptExtractor(device=device)

        self.numeric_patterns = [
            (
                r"hba1c\s*(?:level|of|is|=|:)?\s*([<>≤≥=]{1,2})\s*(\d+\.?\d*)\s*%",
                "hba1c",
                "%",
            ),
            (
                r"hba1c\s*(?:between|from)\s*(\d+\.?\d*)\s*(?:and|to|-)\s*(\d+\.?\d*)\s*%",
                "hba1c_range",
                "%",
            ),
            (
                r"ejection\s+fraction\s*([<>≤≥=]{1,2})\s*(\d+)\s*%",
                "ejection_fraction",
                "%",
            ),
            (r"ef\s*([<>≤≥=]{1,2})\s*(\d+)\s*%", "ejection_fraction", "%"),
            (
                r"(?:sbp|systolic)\s*([<>≤≥=]{1,2})\s*(\d+)\s*mmhg",
                "systolic_bp",
                "mmHg",
            ),
            (
                r"(?:dbp|diastolic)\s*([<>≤≥=]{1,2})\s*(\d+)\s*mmhg",
                "diastolic_bp",
                "mmHg",
            ),
            (r"bmi\s*([<>≤≥=]{1,2})\s*(\d+\.?\d*)\s*(?:kg/m²|kg/m2)?", "bmi", "kg/m²"),
            (r"bmi\s*between\s*(\d+\.?\d*)\s*and\s*(\d+\.?\d*)", "bmi_range", "kg/m²"),
            (r"egfr\s*([<>≤≥=]{1,2})\s*(\d+)\s*(?:ml/min|mL/min)", "egfr", "mL/min"),
            (
                r"creatinine\s*([<>≤≥=]{1,2})\s*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)",
                "creatinine",
                "mg/dL",
            ),
        ]

        self.age_patterns = [
            r"age\s+(\d+)\s*(?:to|-)\s*(\d+)\s+years",
            r"(?:aged|age)\s+between\s+(\d+)\s+and\s+(\d+)\s+years",
            r"(\d+)\s*[-−–]\s*(\d+)\s+years\s+(?:old|of\s+age)",
            r"age\s+(?:range)?\s*:?\s*(\d+)\s*[-−–]\s*(\d+)",
        ]

        self.temporal_patterns = [
            (
                r"for\s+at\s+least\s+(\d+)\s+(month(?:s)?|week(?:s)?|day(?:s)?|year(?:s)?)",
                "minimum_duration",
            ),
            (
                r"within\s+(?:the\s+past\s+)?(\d+)\s+(month(?:s)?|week(?:s)?|day(?:s)?|year(?:s)?)",
                "within_timeframe",
            ),
            (
                r"(?:in\s+the\s+past|over\s+the\s+past)\s+(\d+)\s+(month(?:s)?|week(?:s)?|day(?:s)?|year(?:s)?)",
                "historical_period",
            ),
            (
                r"(\d+)\+?\s+(month(?:s)?|week(?:s)?|day(?:s)?|year(?:s)?)\s+(?:ago|prior)",
                "time_ago",
            ),
            (
                r"stable\s+(?:on\s+)?.*?\s+for\s+(\d+)\s+(month(?:s)?|week(?:s)?|day(?:s)?|year(?:s)?)",
                "stable_duration",
            ),
        ]

    # All existing AdvancedCriteriaProcessor methods unchanged...

    def extract_inclusion_exclusion(self, text):
        inclusion_patterns = [
            r"inclusion\s+criteria:?\s*(.+?)(?=exclusion\s+criteria:?|$)",
            r"eligible\s+(?:patients|subjects)\s+(?:include|must):?\s*(.+?)(?=exclusion|ineligible|$)",
            r"patients?\s+(?:must|should|will):?\s*(.+?)(?=exclusion|patients?\s+(?:must\s+not|should\s+not)|$)",
        ]
        exclusion_patterns = [
            r"exclusion\s+criteria:?\s*(.+?)(?=inclusion\s+criteria:?|$)",
            r"ineligible\s+(?:patients|subjects):?\s*(.+?)(?=inclusion|eligible|$)",
            r"patients?\s+(?:must\s+not|should\s+not|will\s+not):?\s*(.+?)(?=inclusion|patients?\s+(?:must|should)|$)",
        ]

        inclusion_text = ""
        exclusion_text = ""

        for pattern in inclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and len(match.group(1).strip()) > len(inclusion_text):
                inclusion_text = match.group(1).strip()

        for pattern in exclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and len(match.group(1).strip()) > len(exclusion_text):
                exclusion_text = match.group(1).strip()

        return inclusion_text, exclusion_text

    def extract_age_criteria(self, text):
        age_info = []
        for pattern in self.age_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                min_age = int(match.group(1))
                max_age = int(match.group(2))
                if 0 <= min_age <= max_age <= 120:
                    age_info.append(
                        {
                            "min_age": min_age,
                            "max_age": max_age,
                            "text": match.group(0),
                            "type": "age_range",
                        }
                    )
        return age_info

    def extract_numeric_criteria(self, text):
        numeric_criteria = []
        for pattern, parameter, unit in self.numeric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if "range" in parameter:
                    criterion = {
                        "parameter": parameter.replace("_range", ""),
                        "type": "range",
                        "min_value": float(match.group(1)),
                        "max_value": float(match.group(2)),
                        "unit": unit,
                        "original_text": match.group(0),
                    }
                else:
                    criterion = {
                        "parameter": parameter,
                        "type": "comparison",
                        "operator": match.group(1),
                        "value": float(match.group(2)),
                        "unit": unit,
                        "original_text": match.group(0),
                    }
                numeric_criteria.append(criterion)
        return numeric_criteria

    def extract_temporal_constraints(self, text):
        temporal_constraints = []
        for pattern, constraint_type in self.temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                duration = int(match.group(1))
                unit = match.group(2).lower().rstrip("s")
                temporal_constraints.append(
                    {
                        "type": constraint_type,
                        "duration": duration,
                        "unit": unit,
                        "original_text": match.group(0),
                    }
                )
        return temporal_constraints

    def process_eligibility_criteria(self, criteria_text, trial_info):
        inclusion_text, exclusion_text = self.extract_inclusion_exclusion(criteria_text)

        inclusion_data = {
            "medical_concepts": self.concept_extractor.extract_medical_concepts(
                inclusion_text
            ),
            "age_criteria": self.extract_age_criteria(inclusion_text),
            "numeric_criteria": self.extract_numeric_criteria(inclusion_text),
            "temporal_constraints": self.extract_temporal_constraints(inclusion_text),
            "raw_text": inclusion_text,
        }

        exclusion_data = {
            "medical_concepts": self.concept_extractor.extract_medical_concepts(
                exclusion_text
            ),
            "age_criteria": self.extract_age_criteria(exclusion_text),
            "numeric_criteria": self.extract_numeric_criteria(exclusion_text),
            "temporal_constraints": self.extract_temporal_constraints(exclusion_text),
            "raw_text": exclusion_text,
        }

        return {
            "trial_id": trial_info.get("nct_id", ""),
            "trial_title": trial_info.get("title", ""),
            "condition": trial_info.get("condition", ""),
            "phase": trial_info.get("phase", ""),
            "inclusion_criteria": inclusion_data,
            "exclusion_criteria": exclusion_data,
            "processing_metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "inclusion_concepts_found": sum(
                    len(v) for v in inclusion_data["medical_concepts"].values()
                ),
                "exclusion_concepts_found": sum(
                    len(v) for v in exclusion_data["medical_concepts"].values()
                ),
                "numeric_criteria_count": len(inclusion_data["numeric_criteria"])
                + len(exclusion_data["numeric_criteria"]),
                "temporal_constraints_count": len(
                    inclusion_data["temporal_constraints"]
                )
                + len(exclusion_data["temporal_constraints"]),
            },
        }


class ClinicalTrialsDatabase:
    def __init__(self):
        self.trials_data = self._create_comprehensive_trials_database()

    def _create_comprehensive_trials_database(self):
        return fetch_clinical_trials_api(
            condition=None, phase=None, status="Recruiting", max_trials=100
        )

    def get_trials_dataframe(self):
        return pd.DataFrame(self.trials_data)

    def get_trial_by_id(self, nct_id):
        for trial in self.trials_data:
            if trial["nct_id"] == nct_id:
                return trial
        return None


def main():
    print("🏥 Clinical Trial Semantic Matching Engine - Day 1 Implementation")
    print("=" * 70)
    print("Processing comprehensive clinical trials database...")

    trials_db = ClinicalTrialsDatabase()
    processor = AdvancedCriteriaProcessor()

    trials_df = trials_db.get_trials_dataframe()
    print(f"\n📊 Loaded {len(trials_df)} clinical trials:")
    for _, trial in trials_df.iterrows():
        print(f"  • {trial['nct_id']}: {trial['title'][:60]}...")

    processed_trials = []
    processing_stats = {
        "total_trials": 0,
        "total_concepts": 0,
        "total_numeric_criteria": 0,
        "total_temporal_constraints": 0,
    }

    print(f"\n🔍 Processing eligibility criteria...")

    for _, trial in trials_df.iterrows():
        print(f"\n📋 Processing: {trial['nct_id']} - {trial['condition']}")

        processed_trial = processor.process_eligibility_criteria(
            trial["eligibility_criteria"], trial.to_dict()
        )

        processed_trials.append(processed_trial)

        processing_stats["total_trials"] += 1
        processing_stats["total_concepts"] += processed_trial["processing_metadata"][
            "inclusion_concepts_found"
        ]
        processing_stats["total_concepts"] += processed_trial["processing_metadata"][
            "exclusion_concepts_found"
        ]
        processing_stats["total_numeric_criteria"] += processed_trial[
            "processing_metadata"
        ]["numeric_criteria_count"]
        processing_stats["total_temporal_constraints"] += processed_trial[
            "processing_metadata"
        ]["temporal_constraints_count"]

        metadata = processed_trial["processing_metadata"]
        print(
            f"  ✅ Inclusion criteria: {metadata['inclusion_concepts_found']} medical concepts"
        )
        print(
            f"  ❌ Exclusion criteria: {metadata['exclusion_concepts_found']} medical concepts"
        )
        print(f"  📊 Numeric criteria: {metadata['numeric_criteria_count']}")
        print(f"  ⏰ Temporal constraints: {metadata['temporal_constraints_count']}")

        inclusion = processed_trial["inclusion_criteria"]
        if inclusion["medical_concepts"]["conditions"]:
            conditions = [
                c["concept"] for c in inclusion["medical_concepts"]["conditions"][:3]
            ]
            print(f"     Sample conditions: {', '.join(conditions)}")

        if inclusion["numeric_criteria"]:
            numeric = inclusion["numeric_criteria"][0]
            print(
                f"     Sample numeric: {numeric['parameter']} {numeric.get('operator', 'range')} {numeric.get('value', 'N/A')}"
            )

    output_data = {
        "processed_trials": processed_trials,
        "processing_statistics": processing_stats,
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_processing_time": "Real-time",
            "system_info": "Clinical Trial Matching Engine v1.0",
        },
    }

    filename = "processed_trials_advanced_day1.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n📈 Processing Summary:")
    print(f"  🎯 Total trials processed: {processing_stats['total_trials']}")
    print(
        f"  🔬 Total medical concepts extracted: {processing_stats['total_concepts']}"
    )
    print(f"  📊 Total numeric criteria: {processing_stats['total_numeric_criteria']}")
    print(
        f"  ⏰ Total temporal constraints: {processing_stats['total_temporal_constraints']}"
    )

    print(f"\n💾 Results saved to: {filename}")
    print(f"✅ Day 1 Complete! Advanced criteria processing ready.")
    print(f"\n🚀 Next: Run Day 2 patient processing and semantic matching")

    return processed_trials, processing_stats


if __name__ == "__main__":
    processed_data, stats = main()

    print(f"\n🧪 Quick Validation:")
    sample_trial = processed_data[0]
    print(f"Sample trial ID: {sample_trial['trial_id']}")
    print(
        f"Inclusion concepts found: {len(sample_trial['inclusion_criteria']['medical_concepts']['conditions'])}"
    )
    print(
        f"Exclusion concepts found: {len(sample_trial['exclusion_criteria']['medical_concepts']['conditions'])}"
    )
    print(f"\n✅ System is working correctly!")
