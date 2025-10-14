#!/usr/bin/env python3
"""
Advanced Implementation: SNOMED CT Integration
Extends the basic system with full SNOMED CT medical terminology
"""

import csv
import json
import os
import requests
from typing import Dict, List, Optional
from collections import OrderedDict
import sqlite3
import sys


class SNOMEDCTIntegrator:
    """SNOMED CT integration for standardized medical terminology"""

    def __init__(self, snomed_db_path="snomed_ct.db"):
        self.db_path = snomed_db_path
        # Define paths to your actual SNOMED CT data files
        self.snomed_concept_path = "SnomedCT_InternationalRF2_PRODUCTION_20250901T120000Z/Snapshot/Terminology/sct2_Concept_Snapshot_INT_20250901.txt"
        self.snomed_relationship_path = "SnomedCT_InternationalRF2_PRODUCTION_20250901T120000Z/Snapshot/Terminology/sct2_Relationship_Snapshot_INT_20250901.txt"
        self.snomed_description_path = "SnomedCT_InternationalRF2_PRODUCTION_20250901T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20250901.txt"

        # Ensure the data paths are set before proceeding
        self.validate_snomed_paths()

        self.concept_cache = {}

        # Initialize SNOMED CT database connection
        self.conn = sqlite3.connect(snomed_db_path)
        self.setup_snomed_tables()

    def setup_snomed_tables(self):
        """Validate SNOMED CT data paths"""
        if (
            not self.snomed_concept_path
            or not self.snomed_relationship_path
            or not self.snomed_description_path
        ):
            raise ValueError(
                "SNOMED CT data paths are not properly configured.  Please update the paths in the SNOMEDCTIntegrator initialization."
            )

        for path in [
            self.snomed_concept_path,
            self.snomed_relationship_path,
            self.snomed_description_path,
        ]:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"SNOMED CT data file not found: {path}.  Please check the file path and ensure the file exists."
                )

    def validate_snomed_paths(self):
        """Validate SNOMED CT data paths"""
        if (
            not self.snomed_concept_path
            or not self.snomed_relationship_path
            or not self.snomed_description_path
        ):
            raise ValueError(
                "SNOMED CT data paths are not properly configured. Please update the paths in the SNOMEDCTIntegrator initialization."
            )

    def setup_snomed_tables(self):
        """Create SNOMED CT concept tables"""
        cursor = self.conn.cursor()

        # Concepts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS snomed_concepts (
                concept_id TEXT PRIMARY KEY,
                fully_specified_name TEXT,
                preferred_term TEXT,
                definition TEXT,
                active BOOLEAN
            )
        """
        )

        # Relationships table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS snomed_relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                destination_id TEXT,
                type_id TEXT,
                active BOOLEAN,
                FOREIGN KEY (source_id) REFERENCES snomed_concepts (concept_id),
                FOREIGN KEY (destination_id) REFERENCES snomed_concepts (concept_id)
            )
        """
        )

        # Descriptions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS snomed_descriptions (
                id TEXT PRIMARY KEY,
                concept_id TEXT,
                term TEXT,
                type_id TEXT,
                active BOOLEAN,
                FOREIGN KEY (concept_id) REFERENCES snomed_concepts (concept_id)
            )
        """
        )

        self.conn.commit()

    def load_sample_snomed_data(self):
        """REMOVED: Load sample SNOMED CT data for diabetes-related concepts"""
        sample_concepts = [
            {
                "concept_id": "73211009",
                "fully_specified_name": "Diabetes mellitus (disorder)",
                "preferred_term": "Diabetes mellitus",
                "synonyms": ["diabetes", "diabetic condition", "DM"],
            },
            {
                "concept_id": "44054006",
                "fully_specified_name": "Type 2 diabetes mellitus (disorder)",
                "preferred_term": "Type 2 diabetes mellitus",
                "synonyms": [
                    "type 2 diabetes",
                    "T2DM",
                    "non-insulin dependent diabetes",
                ],
            },
            {
                "concept_id": "38341003",
                "fully_specified_name": "Hypertensive disorder (disorder)",
                "preferred_term": "Hypertension",
                "synonyms": ["high blood pressure", "HTN", "hypertensive disease"],
            },
            {
                "concept_id": "84114007",
                "fully_specified_name": "Heart failure (disorder)",
                "preferred_term": "Heart failure",
                "synonyms": [
                    "cardiac failure",
                    "HF",
                    "congestive heart failure",
                    "CHF",
                ],
            },
            {
                "concept_id": "22298006",
                "fully_specified_name": "Myocardial infarction (disorder)",
                "preferred_term": "Myocardial infarction",
                "synonyms": ["heart attack", "MI", "acute MI", "STEMI", "NSTEMI"],
            },
        ]

        cursor = self.conn.cursor()

        for concept in sample_concepts:
            # Insert concept
            cursor.execute(
                """
                INSERT OR REPLACE INTO snomed_concepts 
                (concept_id, fully_specified_name, preferred_term, active)
                VALUES (?, ?, ?, ?)
            """,
                (
                    concept["concept_id"],
                    concept["fully_specified_name"],
                    concept["preferred_term"],
                    True,
                ),
            )

            # Insert synonyms as descriptions
            for synonym in concept["synonyms"]:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO snomed_descriptions
                    (id, concept_id, term, type_id, active)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        f"{concept['concept_id']}_{len(synonym)}",
                        concept["concept_id"],
                        synonym,
                        "900000000000013009",
                        True,
                    ),
                )  # Synonym type

        self.conn.commit()
        print(f"✅ Loaded {len(sample_concepts)} SNOMED CT concepts")

    def load_snomed_data_from_files(self):
        """Load SNOMED CT data from the specified data files."""

        # Increase CSV field size limit for large SNOMED fields.
        # This is necessary because some description fields can be very large.
        max_int = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)

        print("Loading SNOMED data from files...")
        self.load_snomed_concepts()
        self.load_snomed_relationships()
        self.load_snomed_descriptions()
        print("SNOMED data loading complete.")

    def load_snomed_concepts(self):
        """Load SNOMED CT concepts from file"""
        cursor = self.conn.cursor()
        batch_size = 500
        batch = []
        with open(self.snomed_concept_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                batch.append(
                    (
                        row["id"],
                        row["definitionStatusId"],
                        row["active"] == "1",
                    )
                )

                if len(batch) >= batch_size:
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO snomed_concepts 
                        (concept_id, definition, active) 
                        VALUES (?, ?, ?)
                    """,
                        batch,
                    )
                    batch = []
        if batch:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO snomed_concepts 
                (concept_id, definition, active) 
                VALUES (?, ?, ?)
            """,
                batch,
            )
        self.conn.commit()
        print("✅ Loaded SNOMED CT concepts from file")

    def load_snomed_relationships(self):
        """Load SNOMED CT relationships from file"""
        cursor = self.conn.cursor()
        batch_size = 500
        batch = []
        with open(self.snomed_relationship_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                batch.append(
                    (
                        row["id"],
                        row["sourceId"],
                        row["destinationId"],
                        row["typeId"],
                        row["active"],
                    )
                )
                if len(batch) >= batch_size:
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO snomed_relationships 
                        (id, source_id, destination_id, type_id, active) 
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    batch = []
        if batch:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO snomed_relationships 
                (id, source_id, destination_id, type_id, active) 
                VALUES (?, ?, ?, ?, ?)
            """,
                batch,
            )
        self.conn.commit()
        print("✅ Loaded SNOMED CT relationships from file")

    def load_snomed_descriptions(self):
        """Load SNOMED CT descriptions from file and populate preferred terms"""
        cursor = self.conn.cursor()
        batch_size = 500
        batch = []
        preferred_terms = {}
        fsn_terms = {}

        with open(self.snomed_description_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                description_id = row["id"]
                concept_id = row["conceptId"]
                term = row["term"]
                type_id = row["typeId"]
                active = row["active"] == "1"

                if active:
                    # Store Fully Specified Name
                    if type_id == "900000000000003001":  # Fully specified name
                        fsn_terms[concept_id] = term
                    # Store a preferred synonym (take the first active one encountered)
                    elif type_id == "900000000000013009":  # Synonym
                        if concept_id not in preferred_terms:
                            preferred_terms[concept_id] = term

                batch.append((description_id, concept_id, term, type_id, active))

                if len(batch) >= batch_size:
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO snomed_descriptions 
                        (id, concept_id, term, type_id, active) 
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    batch = []
        if batch:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO snomed_descriptions 
                (id, concept_id, term, type_id, active) 
                VALUES (?, ?, ?, ?, ?)
            """,
                batch,
            )
        self.conn.commit()
        print("✅ Loaded SNOMED CT descriptions from file")

        # Update FSN and preferred terms in concepts table
        update_batch = []
        all_concepts = set(fsn_terms.keys()) | set(preferred_terms.keys())
        for concept_id in all_concepts:
            fsn = fsn_terms.get(concept_id, "")
            # Fallback to FSN if no preferred synonym is found
            pt = preferred_terms.get(concept_id, fsn)
            update_batch.append((fsn, pt, concept_id))

        if update_batch:
            cursor.executemany(
                "UPDATE snomed_concepts SET fully_specified_name = ?, preferred_term = ? WHERE concept_id = ?",
                update_batch,
            )
        self.conn.commit()
        print("✅ Updated FSN and preferred terms in concepts table")

    def find_concept_by_term(self, search_term: str) -> Optional[Dict]:
        """Find SNOMED CT concept by search term"""
        cursor = self.conn.cursor()

        # Search in preferred terms and descriptions
        cursor.execute(
            """
            SELECT DISTINCT c.concept_id, c.fully_specified_name, c.preferred_term
            FROM snomed_concepts c
            LEFT JOIN snomed_descriptions d ON c.concept_id = d.concept_id
            WHERE c.preferred_term LIKE ? OR d.term LIKE ?
            AND c.active = 1
            ORDER BY 
                CASE 
                    WHEN c.preferred_term = ? THEN 1
                    WHEN d.term = ? THEN 2
                    WHEN c.preferred_term LIKE ? THEN 3
                    ELSE 4
                END
            LIMIT 5
        """,
            (
                f"%{search_term}%",
                f"%{search_term}%",
                search_term,
                search_term,
                f"{search_term}%",
            ),
        )

        results = cursor.fetchall()

        if results:
            return {
                "concept_id": results[0][0],
                "fully_specified_name": results[0][1],
                "preferred_term": results[0][2],
                "similarity_score": (
                    1.0 if results[0][2].lower() == search_term.lower() else 0.8
                ),
            }

        return None

    def get_concept_hierarchy(self, concept_id: str) -> List[Dict]:
        """Get parent concepts (IS-A relationships)"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT c.concept_id, c.preferred_term
            FROM snomed_concepts c
            JOIN snomed_relationships r ON c.concept_id = r.destination_id
            WHERE r.source_id = ? AND r.type_id = '116680003' AND r.active = 1
        """,
            (concept_id,),
        )

        parents = []
        for row in cursor.fetchall():
            parents.append(
                {"concept_id": row[0], "preferred_term": row[1], "relationship": "IS-A"}
            )

        return parents

    def semantic_similarity(self, term1: str, term2: str) -> float:
        """Calculate semantic similarity using SNOMED CT."""

        concept1 = self.find_concept_by_term(term1)

        concept2 = self.find_concept_by_term(term2)

        if not concept1 or not concept2:
            return 0.0

        # Exact concept match
        if concept1["concept_id"] == concept2["concept_id"]:
            return 1.0

        # Check if concepts are in same hierarchy
        hierarchy1 = self.get_concept_hierarchy(concept1["concept_id"])
        hierarchy2 = self.get_concept_hierarchy(concept2["concept_id"])

        # Check for hierarchical relationships
        hierarchy1_ids = [h["concept_id"] for h in hierarchy1]
        hierarchy2_ids = [h["concept_id"] for h in hierarchy2]

        if concept1["concept_id"] in hierarchy2_ids:
            return 0.9  # concept1 is parent of concept2
        elif concept2["concept_id"] in hierarchy1_ids:
            return 0.9  # concept2 is parent of concept1

        # Check for common ancestors
        common_ancestors = set(hierarchy1_ids) & set(hierarchy2_ids)
        if common_ancestors:
            return 0.7  # Concepts share common ancestor

        # Fallback to string similarity
        return concept1["similarity_score"] * concept2["similarity_score"] * 0.5


# Enhanced medical concept extractor with SNOMED CT
class SNOMEDEnhancedConceptExtractor:  # Enhanced medical concept extractor with SNOMED CT
    """Medical concept extraction enhanced with SNOMED CT"""

    def __init__(self):
        self.snomed = SNOMEDCTIntegrator()
        self.snomed.load_sample_snomed_data()

    def extract_medical_concepts_with_snomed(self, text: str) -> List[Dict]:
        """Extract medical concepts and map to SNOMED CT codes"""
        # Basic NER first (you could enhance this with spaCy medical models)
        import re

        # Simple medical term extraction patterns
        medical_patterns = [
            r"\b(diabetes|diabetic|diabetes mellitus)\b",
            r"\b(hypertension|high blood pressure|htn)\b",
            r"\b(heart failure|cardiac failure|hf|chf)\b",
            r"\b(myocardial infarction|heart attack|mi)\b",
            r"\b(metformin|insulin|lisinopril|atenolol)\b",
        ]

        extracted_concepts = []

        for pattern in medical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(0).lower().strip()

                # Map to SNOMED CT
                snomed_concept = self.snomed.find_concept_by_term(term)

                if snomed_concept:
                    extracted_concepts.append(
                        {
                            "original_term": term,
                            "snomed_concept_id": snomed_concept["concept_id"],
                            "snomed_preferred_term": snomed_concept["preferred_term"],
                            "snomed_fsn": snomed_concept["fully_specified_name"],
                            "confidence": snomed_concept["similarity_score"],
                            "position": match.span(),
                        }
                    )
                else:
                    # Keep original term if no SNOMED mapping found
                    extracted_concepts.append(
                        {
                            "original_term": term,
                            "snomed_concept_id": None,
                            "confidence": 0.5,
                            "position": match.span(),
                        }
                    )

        return extracted_concepts


def demo_snomed_integration():
    """Demonstrate SNOMED CT integration"""
    print("🔬 SNOMED CT Integration Demo")
    print("=" * 40)

    snomed_integrator = SNOMEDCTIntegrator()
    snomed_integrator.load_snomed_data_from_files()

    extractor = SNOMEDEnhancedConceptExtractor()

    # Test patient text
    patient_text = """
    Patient is a 52-year-old male with type 2 diabetes mellitus diagnosed 5 years ago.
    Currently taking metformin for diabetes management. History of hypertension
    controlled with lisinopril. No heart failure or myocardial infarction.
    """

    print("Patient Text:")
    print(patient_text)
    print()

    # Extract concepts with SNOMED mapping
    concepts = extractor.extract_medical_concepts_with_snomed(patient_text)

    print("Extracted SNOMED CT Concepts:")
    for concept in concepts:
        print(f"  • Original: '{concept['original_term']}'")
        if concept["snomed_concept_id"]:
            print(f"    SNOMED ID: {concept['snomed_concept_id']}")
            print(f"    Preferred Term: {concept['snomed_preferred_term']}")
            print(f"    Confidence: {concept['confidence']:.2f}")
        else:
            print(f"    No SNOMED mapping found")
        print()

    # Demonstrate semantic similarity
    print("Semantic Similarity Examples:")
    similarities = [
        ("diabetes", "diabetic"),
        ("heart failure", "cardiac failure"),
        ("myocardial infarction", "heart attack"),
        ("hypertension", "diabetes"),  # Should be low
    ]

    for term1, term2 in similarities:
        similarity = extractor.snomed.semantic_similarity(term1, term2)
        print(f"  '{term1}' ↔ '{term2}': {similarity:.3f}")


if __name__ == "__main__":
    demo_snomed_integration()
