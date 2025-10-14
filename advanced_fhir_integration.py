#!/usr/bin/env python3
"""
Advanced Implementation: FHIR Integration with ClinicalBERT
Real clinical data integration using HL7 FHIR standard and ClinicalBERT NLP embeddings
"""

import json
import requests
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


# ClinicalBERT imports and processor class
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    import torch.nn.functional as F
except ImportError:
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers", "torch"]
    )
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClinicalBERTProcessor:
    """Processor to encode clinical text using ClinicalBERT"""

    _tokenizer_cache = None
    _model_cache = None

    def __init__(
        self, model_name="emilyalsentzer/Bio_ClinicalBERT", pooling_strategy="cls"
    ):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        if (
            not ClinicalBERTProcessor._tokenizer_cache
            or not ClinicalBERTProcessor._model_cache
        ):
            self._load_model_tokenizer()
        self.tokenizer = ClinicalBERTProcessor._tokenizer_cache
        self.model = ClinicalBERTProcessor._model_cache
        self.model.eval()
        self.model.to(DEVICE)

    def _load_model_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        ClinicalBERTProcessor._tokenizer_cache = tokenizer
        ClinicalBERTProcessor._model_cache = model

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, max_length=512, truncation=True, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
            if self.pooling_strategy == "cls":
                sentence_embedding = last_hidden[:, 0, :].cpu().numpy()
            elif self.pooling_strategy == "mean":
                mask = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(last_hidden.size())
                    .float()
                )
                masked_hidden = last_hidden * mask
                summed = masked_hidden.sum(1)
                counts = mask.sum(1)
                mean_hidden = summed / counts
                sentence_embedding = mean_hidden.cpu().numpy()
            else:
                sentence_embedding = last_hidden[:, 0, :].cpu().numpy()
        return sentence_embedding.flatten()

    def semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode_text(text1).reshape(1, -1)
        emb2 = self.encode_text(text2).reshape(1, -1)
        emb1 /= np.linalg.norm(emb1)
        emb2 /= np.linalg.norm(emb2)
        similarity = np.dot(emb1, emb2.T)[0][0]
        return float(similarity)


try:
    from fhir.resources.patient import Patient
    from fhir.resources.observation import Observation
    from fhir.resources.condition import Condition
    from fhir.resources.medicationstatement import MedicationStatement
    from fhir.resources.bundle import Bundle

    FHIR_CLIENT_AVAILABLE = True
    print("✅ FHIR resources library available")
except ImportError:
    print("⚠️ Installing FHIR client library...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "fhir.resources"])
    from fhir.resources.patient import Patient
    from fhir.resources.observation import Observation
    from fhir.resources.condition import Condition
    from fhir.resources.medicationstatement import MedicationStatement
    from fhir.resources.bundle import Bundle

    FHIR_CLIENT_AVAILABLE = True


@dataclass
class ProcessedPatientData:
    """Processed patient data from FHIR resources"""

    patient_id: str
    demographics: Dict
    conditions: List[Dict]
    observations: List[Dict]
    medications: List[Dict]
    clinical_narrative: str


class FHIRIntegrator:
    """FHIR integration for real clinical data processing"""

    def __init__(self, fhir_base_url: str = "http://localhost:8080/fhir"):
        """
        Initialize FHIR integrator

        Common FHIR servers:
        - HAPI FHIR: http://localhost:8080/fhir
        - Public FHIR Server: https://r4.smarthealthit.org
        - Synthea Data: https://synthetichealth.github.io/synthea/
        """
        self.base_url = fhir_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}
        )

    def test_fhir_connection(self) -> bool:
        """Test FHIR server connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/metadata")
            return response.status_code == 200
        except:
            return False

    def create_sample_fhir_patient(self) -> Dict:
        """Create sample FHIR patient for demonstration"""
        patient_data = {
            "resourceType": "Patient",
            "id": "sample-patient-001",
            "name": [
                {"use": "official", "family": "Doe", "given": ["John", "Michael"]}
            ],
            "gender": "male",
            "birthDate": "1971-09-20",  # 52 years old
            "address": [
                {
                    "use": "home",
                    "line": ["123 Main Street"],
                    "city": "Anytown",
                    "state": "CA",
                    "postalCode": "90210",
                    "country": "USA",
                }
            ],
            "telecom": [{"system": "phone", "value": "555-123-4567", "use": "mobile"}],
        }

        # Sample conditions
        conditions = [
            {
                "resourceType": "Condition",
                "id": "condition-diabetes-001",
                "subject": {"reference": "Patient/sample-patient-001"},
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "44054006",
                            "display": "Type 2 diabetes mellitus",
                        }
                    ]
                },
                "onsetDateTime": "2018-09-20",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                        }
                    ]
                },
            },
            {
                "resourceType": "Condition",
                "id": "condition-hypertension-001",
                "subject": {"reference": "Patient/sample-patient-001"},
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "38341003",
                            "display": "Hypertensive disorder",
                        }
                    ]
                },
                "onsetDateTime": "2019-03-15",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                        }
                    ]
                },
            },
        ]

        # Sample observations (lab values)
        observations = [
            {
                "resourceType": "Observation",
                "id": "obs-hba1c-001",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "laboratory",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "4548-4",
                            "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
                        }
                    ]
                },
                "subject": {"reference": "Patient/sample-patient-001"},
                "effectiveDateTime": "2025-09-15",
                "valueQuantity": {
                    "value": 8.4,
                    "unit": "%",
                    "system": "http://unitsofmeasure.org",
                    "code": "%",
                },
            },
            {
                "resourceType": "Observation",
                "id": "obs-bp-001",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "vital-signs",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "85354-9",
                            "display": "Blood pressure panel with all children optional",
                        }
                    ]
                },
                "subject": {"reference": "Patient/sample-patient-001"},
                "effectiveDateTime": "2025-09-15",
                "component": [
                    {
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "8480-6",
                                    "display": "Systolic blood pressure",
                                }
                            ]
                        },
                        "valueQuantity": {
                            "value": 128,
                            "unit": "mmHg",
                            "system": "http://unitsofmeasure.org",
                            "code": "mm[Hg]",
                        },
                    },
                    {
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "8462-4",
                                    "display": "Diastolic blood pressure",
                                }
                            ]
                        },
                        "valueQuantity": {
                            "value": 82,
                            "unit": "mmHg",
                            "system": "http://unitsofmeasure.org",
                            "code": "mm[Hg]",
                        },
                    },
                ],
            },
        ]

        # Sample medications
        medications = [
            {
                "resourceType": "MedicationStatement",
                "id": "med-metformin-001",
                "status": "active",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "6809",
                            "display": "Metformin",
                        }
                    ]
                },
                "subject": {"reference": "Patient/sample-patient-001"},
                "effectiveDateTime": "2023-01-15",
                "dosage": [
                    {
                        "text": "1000 mg twice daily",
                        "timing": {
                            "repeat": {"frequency": 2, "period": 1, "periodUnit": "d"}
                        },
                        "doseAndRate": [
                            {
                                "doseQuantity": {
                                    "value": 1000,
                                    "unit": "mg",
                                    "system": "http://unitsofmeasure.org",
                                    "code": "mg",
                                }
                            }
                        ],
                    }
                ],
            }
        ]

        return {
            "patient": patient_data,
            "conditions": conditions,
            "observations": observations,
            "medications": medications,
        }

    def process_fhir_patient(self, fhir_bundle: Dict) -> ProcessedPatientData:
        """Process FHIR bundle into structured patient data"""

        patient_data = None
        conditions = []
        observations = []
        medications = []

        # If it's a single resource bundle, wrap in list
        if "resourceType" in fhir_bundle and fhir_bundle["resourceType"] == "Patient":
            resources = [fhir_bundle]
        elif "entry" in fhir_bundle:
            resources = [entry["resource"] for entry in fhir_bundle["entry"]]
        else:
            # Assume it's our sample format
            resources = (
                [fhir_bundle["patient"]]
                + fhir_bundle["conditions"]
                + fhir_bundle["observations"]
                + fhir_bundle["medications"]
            )

        # Process each resource
        for resource in resources:
            resource_type = resource.get("resourceType", "")

            if resource_type == "Patient":
                patient_data = self._process_patient_resource(resource)
            elif resource_type == "Condition":
                conditions.append(self._process_condition_resource(resource))
            elif resource_type == "Observation":
                observations.append(self._process_observation_resource(resource))
            elif resource_type == "MedicationStatement":
                medications.append(self._process_medication_resource(resource))

        # Generate clinical narrative
        clinical_narrative = self._generate_clinical_narrative(
            patient_data, conditions, observations, medications
        )

        return ProcessedPatientData(
            patient_id=patient_data["id"] if patient_data else "unknown",
            demographics=patient_data if patient_data else {},
            conditions=conditions,
            observations=observations,
            medications=medications,
            clinical_narrative=clinical_narrative,
        )

    def _process_patient_resource(self, patient: Dict) -> Dict:
        """Process FHIR Patient resource"""
        demographics = {
            "id": patient.get("id", ""),
            "gender": patient.get("gender", "unknown"),
        }

        # Extract name
        if "name" in patient and patient["name"]:
            name_data = patient["name"][0]
            full_name = ""
            if "given" in name_data:
                full_name = " ".join(name_data["given"])
            if "family" in name_data:
                full_name += " " + name_data["family"]
            demographics["name"] = full_name.strip()

        # Calculate age from birth date
        if "birthDate" in patient:
            birth_date = datetime.strptime(patient["birthDate"], "%Y-%m-%d").date()
            today = date.today()
            age = (
                today.year
                - birth_date.year
                - ((today.month, today.day) < (birth_date.month, birth_date.day))
            )
            demographics["age"] = age
            demographics["birth_date"] = patient["birthDate"]

        return demographics

    def _process_condition_resource(self, condition: Dict) -> Dict:
        """Process FHIR Condition resource"""
        condition_data = {"id": condition.get("id", ""), "status": "unknown"}

        # Extract condition name
        if "code" in condition and "coding" in condition["code"]:
            coding = condition["code"]["coding"][0]
            condition_data["name"] = coding.get("display", "Unknown condition")
            condition_data["snomed_code"] = (
                coding.get("code", "")
                if coding.get("system") == "http://snomed.info/sct"
                else ""
            )

        # Extract clinical status
        if "clinicalStatus" in condition and "coding" in condition["clinicalStatus"]:
            condition_data["status"] = condition["clinicalStatus"]["coding"][0].get(
                "code", "unknown"
            )

        # Extract onset date
        if "onsetDateTime" in condition:
            condition_data["onset_date"] = condition["onsetDateTime"]

        return condition_data

    def _process_observation_resource(self, observation: Dict) -> Dict:
        """Process FHIR Observation resource"""
        obs_data = {
            "id": observation.get("id", ""),
            "status": observation.get("status", "unknown"),
        }

        # Extract observation name and LOINC code
        if "code" in observation and "coding" in observation["code"]:
            coding = observation["code"]["coding"][0]
            obs_data["name"] = coding.get("display", "Unknown observation")
            obs_data["loinc_code"] = (
                coding.get("code", "")
                if coding.get("system") == "http://loinc.org"
                else ""
            )

        # Extract date
        if "effectiveDateTime" in observation:
            obs_data["date"] = observation["effectiveDateTime"]

        # Extract value
        if "valueQuantity" in observation:
            value_qty = observation["valueQuantity"]
            obs_data["value"] = value_qty.get("value", 0)
            obs_data["unit"] = value_qty.get("unit", "")

        # Handle component observations (like blood pressure)
        if "component" in observation:
            components = []
            for comp in observation["component"]:
                comp_data = {}
                if "code" in comp and "coding" in comp["code"]:
                    comp_coding = comp["code"]["coding"][0]
                    comp_data["name"] = comp_coding.get("display", "")
                    comp_data["loinc_code"] = comp_coding.get("code", "")

                if "valueQuantity" in comp:
                    comp_value = comp["valueQuantity"]
                    comp_data["value"] = comp_value.get("value", 0)
                    comp_data["unit"] = comp_value.get("unit", "")

                components.append(comp_data)

            obs_data["components"] = components

        return obs_data

    def _process_medication_resource(self, medication: Dict) -> Dict:
        """Process FHIR MedicationStatement resource"""
        med_data = {
            "id": medication.get("id", ""),
            "status": medication.get("status", "unknown"),
        }

        # Extract medication name
        if (
            "medicationCodeableConcept" in medication
            and "coding" in medication["medicationCodeableConcept"]
        ):
            coding = medication["medicationCodeableConcept"]["coding"][0]
            med_data["name"] = coding.get("display", "Unknown medication")
            med_data["rxnorm_code"] = (
                coding.get("code", "")
                if coding.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm"
                else ""
            )

        # Extract dosage information
        if "dosage" in medication and medication["dosage"]:
            dosage = medication["dosage"][0]
            med_data["dosage_text"] = dosage.get("text", "")

            if "doseAndRate" in dosage and dosage["doseAndRate"]:
                dose_info = dosage["doseAndRate"][0]
                if "doseQuantity" in dose_info:
                    dose_qty = dose_info["doseQuantity"]
                    med_data["dose_value"] = dose_qty.get("value", 0)
                    med_data["dose_unit"] = dose_qty.get("unit", "")

        # Extract effective date
        if "effectiveDateTime" in medication:
            med_data["start_date"] = medication["effectiveDateTime"]

        return med_data

    def _generate_clinical_narrative(
        self,
        demographics: Dict,
        conditions: List[Dict],
        observations: List[Dict],
        medications: List[Dict],
    ) -> str:
        """Generate clinical narrative text from FHIR data"""

        narrative_parts = []

        # Demographics
        if demographics:
            age = demographics.get("age", "unknown age")
            gender = demographics.get("gender", "unknown gender")
            narrative_parts.append(f"Patient is a {age}-year-old {gender}")

        # Conditions
        if conditions:
            active_conditions = [c for c in conditions if c.get("status") == "active"]
            if active_conditions:
                condition_names = [c["name"] for c in active_conditions]
                narrative_parts.append(f"with {', '.join(condition_names)}")

        # Medications
        if medications:
            active_meds = [m for m in medications if m.get("status") == "active"]
            if active_meds:
                med_text = []
                for med in active_meds:
                    med_name = med["name"]
                    dosage = med.get("dosage_text", "")
                    if dosage:
                        med_text.append(f"{med_name} {dosage}")
                    else:
                        med_text.append(med_name)
                narrative_parts.append(f"Currently taking {', '.join(med_text)}")

        # Recent observations
        if observations:
            recent_obs = sorted(
                observations, key=lambda x: x.get("date", ""), reverse=True
            )[:3]
            obs_text = []

            for obs in recent_obs:
                if "components" in obs:  # Blood pressure
                    bp_components = obs["components"]
                    if len(bp_components) == 2:
                        systolic = next(
                            (
                                c
                                for c in bp_components
                                if "systolic" in c.get("name", "").lower()
                            ),
                            None,
                        )
                        diastolic = next(
                            (
                                c
                                for c in bp_components
                                if "diastolic" in c.get("name", "").lower()
                            ),
                            None,
                        )
                        if systolic and diastolic:
                            obs_text.append(
                                f"Blood pressure {systolic['value']}/{diastolic['value']} {systolic.get('unit', 'mmHg')}"
                            )
                else:
                    name = obs.get("name", "").replace(
                        "Hemoglobin A1c/Hemoglobin.total in Blood", "HbA1c"
                    )
                    value = obs.get("value", "")
                    unit = obs.get("unit", "")
                    obs_text.append(f"{name} {value}{unit}")

            if obs_text:
                narrative_parts.append(f"Recent lab values: {', '.join(obs_text)}")

        return ". ".join(narrative_parts) + "."


class FHIREnhancedMatchingSystem:
    """Enhanced matching system with FHIR integration using ClinicalBERT"""

    def __init__(self):
        self.fhir_integrator = FHIRIntegrator()
        self.clinicalbert = ClinicalBERTProcessor()

    def process_fhir_patient_for_matching(self, fhir_data: Dict) -> Dict:
        """Process FHIR patient data for clinical trial matching"""

        # Process FHIR resources
        processed_patient = self.fhir_integrator.process_fhir_patient(fhir_data)

        # Convert to format compatible with matching system
        patient_for_matching = {
            "patient_id": processed_patient.patient_id,
            "demographics": processed_patient.demographics,
            "lab_values": self._extract_lab_values_from_fhir(
                processed_patient.observations
            ),
            "medical_concepts": self._extract_concepts_from_fhir(
                processed_patient.conditions, processed_patient.medications
            ),
            "clinical_text": processed_patient.clinical_narrative,
            "fhir_source": True,
        }

        # Optionally, encode clinical narrative text here for further semantic matching
        patient_for_matching["clinical_text_embedding"] = self.clinicalbert.encode_text(
            processed_patient.clinical_narrative
        )

        return patient_for_matching

    def _extract_lab_values_from_fhir(self, observations: List[Dict]) -> Dict:
        """Extract lab values from FHIR observations"""
        lab_values = {}

        for obs in observations:
            loinc_code = obs.get("loinc_code", "")

            # Map LOINC codes to our lab value names
            if loinc_code == "4548-4":  # HbA1c
                lab_values["hba1c"] = obs.get("value", 0)
            elif loinc_code == "2160-0":  # Creatinine
                lab_values["creatinine"] = obs.get("value", 0)
            elif "components" in obs:  # Blood pressure
                for comp in obs["components"]:
                    if comp.get("loinc_code") == "8480-6":  # Systolic BP
                        lab_values["systolic_bp"] = comp.get("value", 0)
                    elif comp.get("loinc_code") == "8462-4":  # Diastolic BP
                        lab_values["diastolic_bp"] = comp.get("value", 0)

        return lab_values

    def _extract_concepts_from_fhir(
        self, conditions: List[Dict], medications: List[Dict]
    ) -> Dict:
        """Extract medical concepts from FHIR conditions and medications"""
        medical_concepts = {"conditions": [], "medications": [], "measurements": []}

        # Process conditions
        for condition in conditions:
            if condition.get("status") == "active":
                medical_concepts["conditions"].append(
                    {
                        "concept": condition.get("name", "").lower(),
                        "snomed_code": condition.get("snomed_code", ""),
                        "is_negated": False,
                        "confidence": 1.0,
                    }
                )

        # Process medications
        for medication in medications:
            if medication.get("status") == "active":
                medical_concepts["medications"].append(
                    {
                        "concept": medication.get("name", "").lower(),
                        "rxnorm_code": medication.get("rxnorm_code", ""),
                        "is_negated": False,
                        "confidence": 1.0,
                    }
                )

        return medical_concepts


def demo_fhir_integration():
    """Demonstrate FHIR integration"""
    print("🏥 FHIR Integration Demo with ClinicalBERT")
    print("=" * 30)

    # Initialize FHIR integrator and matching system
    fhir_integrator = FHIRIntegrator()
    matching_system = FHIREnhancedMatchingSystem()

    # Create sample FHIR patient data
    print("📋 Creating sample FHIR patient data...")
    sample_fhir_data = fhir_integrator.create_sample_fhir_patient()

    # Process FHIR data
    print("🔄 Processing FHIR resources...")
    processed_patient = fhir_integrator.process_fhir_patient(sample_fhir_data)

    # Display processed patient information
    print("\n👤 Processed Patient Information:")
    print("-" * 35)
    print(f"Patient ID: {processed_patient.patient_id}")
    print(f"Demographics: {processed_patient.demographics}")

    print("\n🏥 Conditions:")
    for condition in processed_patient.conditions:
        status_emoji = "🔴" if condition["status"] == "active" else "⚪"
        print(
            f"  {status_emoji} {condition['name']} (SNOMED: {condition.get('snomed_code', 'N/A')})"
        )

    print("\n🔬 Observations:")
    for obs in processed_patient.observations:
        if "components" in obs:
            print(f"  📊 {obs['name']}:")
            for comp in obs["components"]:
                print(
                    f"    • {comp['name']}: {comp['value']} {comp['unit']} (LOINC: {comp.get('loinc_code', 'N/A')})"
                )
        else:
            print(
                f"  📊 {obs['name']}: {obs.get('value', 'N/A')} {obs.get('unit', '')} (LOINC: {obs.get('loinc_code', 'N/A')})"
            )

    print("\n💊 Medications:")
    for med in processed_patient.medications:
        status_emoji = "✅" if med["status"] == "active" else "⏸️"
        print(
            f"  {status_emoji} {med['name']} - {med.get('dosage_text', 'No dosage info')} (RxNorm: {med.get('rxnorm_code', 'N/A')})"
        )

    print("\n📄 Generated Clinical Narrative:")
    print("-" * 35)
    print(processed_patient.clinical_narrative)

    # Convert for matching
    print("\n🎯 Converting for Clinical Trial Matching:")
    patient_for_matching = matching_system.process_fhir_patient_for_matching(
        sample_fhir_data
    )

    print(f"Extracted Lab Values: {patient_for_matching['lab_values']}")
    print(
        f"Medical Concepts: {len(patient_for_matching['medical_concepts']['conditions'])} conditions, {len(patient_for_matching['medical_concepts']['medications'])} medications"
    )
    print(
        f"Clinical Text Embedding Vector Length: {len(patient_for_matching['clinical_text_embedding'])}"
    )

    print(
        "\n✅ FHIR integration with ClinicalBERT complete! Patient data ready for matching."
    )

    # Show FHIR resource structure
    print("\n📋 Sample FHIR Resources Structure:")
    print("-" * 35)
    print(
        f"Patient Resource: {json.dumps(sample_fhir_data['patient'], indent=2)[:200]}..."
    )
    print(
        f"Condition Resource: {json.dumps(sample_fhir_data['conditions'][0], indent=2)[:200]}..."
    )
    print(
        f"Observation Resource: {json.dumps(sample_fhir_data['observations'][0], indent=2)[:200]}..."
    )


if __name__ == "__main__":
    demo_fhir_integration()
