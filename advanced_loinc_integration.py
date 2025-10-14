#!/usr/bin/env python3
"""
Advanced Implementation: LOINC Integration
Laboratory data standardization using LOINC codes
"""

import json
import sqlite3
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LOINCConcept:
    """LOINC concept data structure"""
    loinc_num: str
    component: str
    property: str
    time_aspct: str
    system: str
    scale_typ: str
    method_typ: str
    shortname: str
    long_common_name: str
    example_units: str

class LOINCIntegrator:
    """LOINC integration for standardized laboratory data"""

    def __init__(self, loinc_db_path="loinc.db"):
        self.db_path = loinc_db_path
        self.conn = sqlite3.connect(loinc_db_path)
        self.setup_loinc_tables()
        self.load_sample_loinc_data()

    def setup_loinc_tables(self):
        """Create LOINC database tables"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loinc_concepts (
                loinc_num TEXT PRIMARY KEY,
                component TEXT,
                property TEXT,
                time_aspct TEXT,
                system TEXT,
                scale_typ TEXT,
                method_typ TEXT,
                shortname TEXT,
                long_common_name TEXT,
                example_units TEXT,
                status TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loinc_synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                loinc_num TEXT,
                synonym TEXT,
                synonym_type TEXT,
                FOREIGN KEY (loinc_num) REFERENCES loinc_concepts (loinc_num)
            )
        """)

        self.conn.commit()

    def load_sample_loinc_data(self):
        """Load sample LOINC data for common lab tests"""
        sample_loinc_data = [
            LOINCConcept(
                loinc_num="4548-4",
                component="Hemoglobin A1c",
                property="MFr",
                time_aspct="Pt",
                system="Bld",
                scale_typ="Qn",
                method_typ="",
                shortname="HbA1c MFr Bld",
                long_common_name="Hemoglobin A1c/Hemoglobin.total in Blood",
                example_units="%"
            ),
            LOINCConcept(
                loinc_num="2339-0", 
                component="Glucose",
                property="MCnc",
                time_aspct="Pt",
                system="Ser/Plas",
                scale_typ="Qn",
                method_typ="",
                shortname="Glucose Ser/Plas",
                long_common_name="Glucose [Mass/volume] in Serum or Plasma",
                example_units="mg/dL"
            ),
            LOINCConcept(
                loinc_num="2160-0",
                component="Creatinine",
                property="MCnc", 
                time_aspct="Pt",
                system="Ser/Plas",
                scale_typ="Qn",
                method_typ="",
                shortname="Creatinine Ser/Plas",
                long_common_name="Creatinine [Mass/volume] in Serum or Plasma",
                example_units="mg/dL"
            ),
            LOINCConcept(
                loinc_num="33747-0",
                component="Estimated GFR",
                property="VRat",
                time_aspct="Pt", 
                system="Ser/Plas/Bld",
                scale_typ="Qn",
                method_typ="MDRD",
                shortname="eGFR MDRD",
                long_common_name="Glomerular filtration rate/1.73 sq M.predicted",
                example_units="mL/min/1.73m2"
            ),
            LOINCConcept(
                loinc_num="10834-0",
                component="Ejection fraction",
                property="VFr",
                time_aspct="Pt",
                system="Heart.ventricle.left",
                scale_typ="Qn",
                method_typ="Echo",
                shortname="LVEF Echo",
                long_common_name="Left ventricular Ejection fraction by 2D echo",
                example_units="%"
            ),
            LOINCConcept(
                loinc_num="8480-6",
                component="Systolic blood pressure",
                property="Pres",
                time_aspct="Pt",
                system="Arterial",
                scale_typ="Qn", 
                method_typ="",
                shortname="Systolic BP",
                long_common_name="Systolic blood pressure",
                example_units="mmHg"
            ),
            LOINCConcept(
                loinc_num="8462-4",
                component="Diastolic blood pressure", 
                property="Pres",
                time_aspct="Pt",
                system="Arterial",
                scale_typ="Qn",
                method_typ="",
                shortname="Diastolic BP", 
                long_common_name="Diastolic blood pressure",
                example_units="mmHg"
            )
        ]

        cursor = self.conn.cursor()

        for loinc in sample_loinc_data:
            cursor.execute("""
                INSERT OR REPLACE INTO loinc_concepts 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                loinc.loinc_num, loinc.component, loinc.property,
                loinc.time_aspct, loinc.system, loinc.scale_typ,
                loinc.method_typ, loinc.shortname, loinc.long_common_name,
                loinc.example_units, "ACTIVE"
            ))

            # Add common synonyms
            synonyms = self._get_synonyms_for_concept(loinc)
            for synonym in synonyms:
                cursor.execute("""
                    INSERT OR REPLACE INTO loinc_synonyms 
                    (loinc_num, synonym, synonym_type) VALUES (?, ?, ?)
                """, (loinc.loinc_num, synonym, "COMMON_NAME"))

        self.conn.commit()
        print(f"✅ Loaded {len(sample_loinc_data)} LOINC concepts")

    def _get_synonyms_for_concept(self, loinc: LOINCConcept) -> List[str]:
        """Generate common synonyms for LOINC concepts"""
        synonyms = []

        if "hemoglobin a1c" in loinc.long_common_name.lower():
            synonyms = ["hba1c", "a1c", "glycated hemoglobin", "hemoglobin a1c"]
        elif "glucose" in loinc.long_common_name.lower():
            synonyms = ["glucose", "blood glucose", "blood sugar", "fasting glucose"]
        elif "creatinine" in loinc.long_common_name.lower():
            synonyms = ["creatinine", "serum creatinine", "cr"]
        elif "glomerular filtration" in loinc.long_common_name.lower():
            synonyms = ["egfr", "estimated gfr", "gfr", "kidney function"]
        elif "ejection fraction" in loinc.long_common_name.lower():
            synonyms = ["ejection fraction", "ef", "lvef", "left ventricular ef"]
        elif "systolic" in loinc.long_common_name.lower():
            synonyms = ["systolic bp", "systolic blood pressure", "sbp"]
        elif "diastolic" in loinc.long_common_name.lower():
            synonyms = ["diastolic bp", "diastolic blood pressure", "dbp"]

        return synonyms

    def find_loinc_by_term(self, search_term: str) -> Optional[LOINCConcept]:
        """Find LOINC code by search term"""
        cursor = self.conn.cursor()

        # Search in component, short name, long name, and synonyms
        cursor.execute("""
            SELECT DISTINCT l.* FROM loinc_concepts l
            LEFT JOIN loinc_synonyms s ON l.loinc_num = s.loinc_num
            WHERE l.component LIKE ? OR l.shortname LIKE ? OR 
                  l.long_common_name LIKE ? OR s.synonym LIKE ?
            ORDER BY 
                CASE 
                    WHEN s.synonym = ? THEN 1
                    WHEN l.component = ? THEN 2
                    WHEN l.shortname LIKE ? THEN 3
                    ELSE 4
                END
            LIMIT 1
        """, tuple([f'%{search_term}%'] * 4 + [search_term, search_term, f'{search_term}%']))

        result = cursor.fetchone()

        if result:
            return LOINCConcept(*result[:-1])  # Exclude status column

        return None

    def extract_lab_values_with_loinc(self, text: str) -> List[Dict]:
        """Extract lab values and map to LOINC codes"""

        # Enhanced patterns with LOINC mapping potential
        lab_patterns = [
            # HbA1c patterns
            (r'(?:hba1c|hemoglobin\s*a1c|a1c)\s*(?:level|is|of|=|:)?\s*(\d+\.?\d*)\s*%?', 'hba1c'),

            # Glucose patterns
            (r'(?:glucose|blood\s*glucose|fasting\s*glucose)\s*(?:level|is|of|=|:)?\s*(\d+)\s*(?:mg/dl|mg/dL)?', 'glucose'),

            # Creatinine patterns
            (r'(?:creatinine|serum\s*creatinine)\s*(?:level|is|of|=|:)?\s*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)?', 'creatinine'),

            # eGFR patterns
            (r'(?:egfr|estimated\s*gfr|gfr)\s*(?:is|of|=|:)?\s*(\d+)\s*(?:ml/min|mL/min)?', 'egfr'),

            # Ejection fraction patterns
            (r'(?:ejection\s*fraction|ef|lvef)\s*(?:is|of|=|:)?\s*(\d+)\s*%?', 'ejection fraction'),

            # Blood pressure patterns
            (r'(?:blood\s*pressure|bp)\s*(?:is|of|=|:)?\s*(\d+)/(\d+)', 'blood pressure'),
            (r'(?:systolic)\s*(?:bp|blood\s*pressure)?\s*(?:is|of|=|:)?\s*(\d+)', 'systolic bp'),
            (r'(?:diastolic)\s*(?:bp|blood\s*pressure)?\s*(?:is|of|=|:)?\s*(\d+)', 'diastolic bp')
        ]

        extracted_values = []

        for pattern, lab_name in lab_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find LOINC code for this lab
                loinc_concept = self.find_loinc_by_term(lab_name)

                if lab_name == 'blood pressure':
                    # Special handling for BP (two values)
                    systolic = int(match.group(1))
                    diastolic = int(match.group(2))

                    # Get LOINC codes for both
                    systolic_loinc = self.find_loinc_by_term('systolic bp')
                    diastolic_loinc = self.find_loinc_by_term('diastolic bp')

                    extracted_values.extend([
                        {
                            'original_text': match.group(0),
                            'lab_name': 'systolic blood pressure',
                            'value': systolic,
                            'unit': 'mmHg',
                            'loinc_code': systolic_loinc.loinc_num if systolic_loinc else None,
                            'loinc_name': systolic_loinc.long_common_name if systolic_loinc else None,
                            'position': match.span()
                        },
                        {
                            'original_text': match.group(0),
                            'lab_name': 'diastolic blood pressure', 
                            'value': diastolic,
                            'unit': 'mmHg',
                            'loinc_code': diastolic_loinc.loinc_num if diastolic_loinc else None,
                            'loinc_name': diastolic_loinc.long_common_name if diastolic_loinc else None,
                            'position': match.span()
                        }
                    ])
                else:
                    # Single value labs
                    value = float(match.group(1))
                    unit = loinc_concept.example_units if loinc_concept else ""

                    extracted_values.append({
                        'original_text': match.group(0),
                        'lab_name': lab_name,
                        'value': value,
                        'unit': unit,
                        'loinc_code': loinc_concept.loinc_num if loinc_concept else None,
                        'loinc_name': loinc_concept.long_common_name if loinc_concept else None,
                        'position': match.span()
                    })

        return extracted_values

    def validate_lab_value(self, loinc_code: str, value: float) -> Dict:
        """Validate lab value against normal ranges (simplified)"""
        # Simplified reference ranges - in production, use comprehensive ranges
        reference_ranges = {
            "4548-4": {"normal_range": (4.0, 5.6), "unit": "%", "name": "HbA1c"},  # HbA1c
            "2339-0": {"normal_range": (70, 99), "unit": "mg/dL", "name": "Glucose"},  # Fasting glucose
            "2160-0": {"normal_range": (0.7, 1.3), "unit": "mg/dL", "name": "Creatinine"},  # Creatinine
            "33747-0": {"normal_range": (90, 120), "unit": "mL/min/1.73m2", "name": "eGFR"},  # eGFR
            "10834-0": {"normal_range": (50, 70), "unit": "%", "name": "Ejection Fraction"},  # LVEF
            "8480-6": {"normal_range": (90, 120), "unit": "mmHg", "name": "Systolic BP"},  # Systolic BP
            "8462-4": {"normal_range": (60, 80), "unit": "mmHg", "name": "Diastolic BP"}   # Diastolic BP
        }

        if loinc_code not in reference_ranges:
            return {"status": "unknown", "message": "No reference range available"}

        ref_range = reference_ranges[loinc_code]
        min_val, max_val = ref_range["normal_range"]

        if min_val <= value <= max_val:
            return {"status": "normal", "message": f"Within normal range ({min_val}-{max_val} {ref_range['unit']})"}
        elif value < min_val:
            return {"status": "low", "message": f"Below normal range (<{min_val} {ref_range['unit']})"}
        else:
            return {"status": "high", "message": f"Above normal range (>{max_val} {ref_range['unit']})"}

def demo_loinc_integration():
    """Demonstrate LOINC integration"""
    print("🧪 LOINC Integration Demo")
    print("=" * 35)

    loinc_integrator = LOINCIntegrator()

    # Test patient text with lab values
    patient_text = """
    Patient lab results from recent visit:
    - HbA1c level is 8.4%
    - Fasting glucose 165 mg/dL
    - Serum creatinine 1.1 mg/dL
    - eGFR 88 mL/min/1.73m2
    - Blood pressure 128/82 mmHg
    - Ejection fraction 62% on echocardiogram
    """

    print("Patient Text:")
    print(patient_text)
    print()

    # Extract lab values with LOINC mapping
    lab_values = loinc_integrator.extract_lab_values_with_loinc(patient_text)

    print("Extracted Lab Values with LOINC Codes:")
    print("-" * 50)

    for lab in lab_values:
        print(f"Lab: {lab['lab_name']}")
        print(f"  Value: {lab['value']} {lab['unit']}")
        print(f"  LOINC Code: {lab['loinc_code']}")
        print(f"  LOINC Name: {lab['loinc_name']}")

        # Validate against reference ranges
        if lab['loinc_code']:
            validation = loinc_integrator.validate_lab_value(lab['loinc_code'], lab['value'])
            status_emoji = {"normal": "✅", "high": "⚠️", "low": "⚠️", "unknown": "❓"}[validation['status']]
            print(f"  Status: {status_emoji} {validation['message']}")

        print()

    # Demonstrate LOINC lookup
    print("LOINC Lookup Examples:")
    print("-" * 25)
    search_terms = ["hba1c", "creatinine", "ejection fraction", "blood pressure"]

    for term in search_terms:
        loinc = loinc_integrator.find_loinc_by_term(term)
        if loinc:
            print(f"'{term}' → LOINC {loinc.loinc_num}: {loinc.long_common_name}")
        else:
            print(f"'{term}' → No LOINC code found")

if __name__ == "__main__":
    demo_loinc_integration()
