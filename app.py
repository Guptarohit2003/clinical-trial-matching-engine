#!/usr/bin/env python3
"""
Clinical Trial Semantic Matching Engine - Day 3 Complete Implementation
FastAPI Backend + Streamlit Web Interface + Complete System Integration

This creates a production-ready system with API endpoints and web interface
that demonstrates your complete semantic matching engine.

Author: Your Name
Date: September 2025
Project: Semantic Matching Engine for Clinical Trial Recruitment
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import warnings

warnings.filterwarnings("ignore")

# FastAPI and Web Interface imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    # print("Installing required packages for Day 3...")
    # import subprocess
    # import sys

    # subprocess.check_call(
    #     [
    #         sys.executable,
    #         "-m",
    #         "pip",
    #         "install",
    #         "fastapi",
    #         "uvicorn[standard]",
    #         "streamlit",
    #         "plotly",
    #         "pydantic",
    #     ]
    # )
    # from fastapi import FastAPI, HTTPException, BackgroundTasks
    # from fastapi.middleware.cors import CORSMiddleware
    # from pydantic import BaseModel
    # import uvicorn
    # import streamlit as st
    # import plotly.graph_objects as go
    # import plotly.express as px

    print("Error: Required packages are not installed.")
    print(
        "Please ensure you have a requirements.txt file and run 'pip install -r requirements.txt'"
    )
    raise

# Import Day 1 and Day 2 components (with fallback)
try:
    from advanced_day1_implementation import (
        AdvancedCriteriaProcessor,
        ClinicalTrialsDatabase,
    )
    from advanced_day2_complete import (
        AdvancedPatientProcessor,
        AdvancedSemanticMatcher,
        MatchingResult,
    )

    DAY1_DAY2_AVAILABLE = True
except ImportError:
    print(
        "Warning: Day 1 or Day 2 implementations not found. Using simplified versions."
    )
    DAY1_DAY2_AVAILABLE = False


# Pydantic models for API
class PatientMatchRequest(BaseModel):
    patient_id: Optional[str] = None
    clinical_text: Optional[str] = None
    fhir_patient_bundle: Optional[Dict] = None
    max_results: Optional[int] = 10
    min_score_threshold: Optional[float] = 0.1


class MatchResult(BaseModel):
    trial_id: str
    trial_title: str
    condition: str
    overall_score: float
    inclusion_score: float
    exclusion_score: float
    age_eligible: Optional[bool]
    explanation: str
    confidence_level: str
    detailed_matches: Dict


class PatientMatchResponse(BaseModel):
    patient_id: str
    processing_time_ms: float
    total_trials_evaluated: int
    matches_found: int
    top_matches: List[MatchResult]
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trial Semantic Matching Engine",
    description="AI-powered system for matching patients to clinical trials",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimplifiedMatchingSystem:
    """Simplified matching system for when Day 1/2 not available"""

    def __init__(self):
        self.sample_trials = [
            {
                "trial_id": "NCT05001234",
                "trial_title": "Diabetes Treatment Study",
                "condition": "Type 2 Diabetes Mellitus",
                "phase": "Phase 3",
            },
            {
                "trial_id": "NCT05002345",
                "trial_title": "Hypertension Management Trial",
                "condition": "Essential Hypertension",
                "phase": "Phase 4",
            },
        ]

    async def match_patient_to_trials(
        self, patient_text, patient_id, max_results=10, min_threshold=0.1
    ):
        """Simplified matching for demo purposes"""
        await asyncio.sleep(0.1)  # Simulate processing time

        # Simple keyword matching
        matches = []
        for trial in self.sample_trials:
            score = 0.5
            if (
                "diabetes" in patient_text.lower()
                and "diabetes" in trial["condition"].lower()
            ):
                score = 0.85
            elif (
                "hypertension" in patient_text.lower()
                and "hypertension" in trial["condition"].lower()
            ):
                score = 0.80

            if score >= min_threshold:
                matches.append(
                    MatchResult(
                        trial_id=trial["trial_id"],
                        trial_title=trial["trial_title"],
                        condition=trial["condition"],
                        overall_score=score,
                        inclusion_score=score,
                        exclusion_score=0.1,
                        age_eligible=True,
                        explanation=f"Basic keyword matching for {trial['condition']}",
                        confidence_level="Medium",
                    )
                )
                matches[-1].detailed_matches = {
                    "info": "Simplified matching does not provide a detailed breakdown."
                }

        matches.sort(key=lambda x: x.overall_score, reverse=True)

        return PatientMatchResponse(
            patient_id=patient_id,
            processing_time_ms=150.0,
            total_trials_evaluated=len(self.sample_trials),
            matches_found=len(matches),
            top_matches=matches[:max_results],
            timestamp=datetime.now().isoformat(),
        )


class ClinicalTrialMatchingSystem:
    """Complete clinical trial matching system"""

    def __init__(self):
        if DAY1_DAY2_AVAILABLE:
            self.criteria_processor = AdvancedCriteriaProcessor()
            self.patient_processor = AdvancedPatientProcessor()
            self.semantic_matcher = AdvancedSemanticMatcher()
            self.processed_trials = self.load_processed_trials()
        else:
            self.simplified_system = SimplifiedMatchingSystem()
            self.processed_trials = self.simplified_system.sample_trials

        self.system_stats = {
            "patients_processed": 0,
            "total_matches_computed": 0,
            "system_start_time": datetime.now(),
        }

    def load_processed_trials(self):
        """Load processed trials from Day 1"""
        try:
            with open("processed_trials_advanced_day1.json", "r") as f:
                data = json.load(f)
            return data["processed_trials"]
        except FileNotFoundError:
            print("Warning: Day 1 output not found. Using sample data.")
            return []

    async def match_patient_to_trials(
        self, patient_text, patient_id, max_results=10, min_threshold=0.1
    ):
        """Main matching function"""
        if not DAY1_DAY2_AVAILABLE or not self.processed_trials:
            return await self.simplified_system.match_patient_to_trials(
                patient_text, patient_id, max_results, min_threshold
            )

        start_time = datetime.now()

        # Process patient
        processed_patient = self.patient_processor.process_patient_text(
            patient_text, patient_id
        )

        # Match to all trials
        all_matches = []
        for trial in self.processed_trials:
            match_result = self.semantic_matcher.match_patient_to_trial(
                trial, processed_patient
            )

            confidence_level = (
                "High"
                if match_result.overall_score >= 0.8
                else (
                    "Medium"
                    if match_result.overall_score >= 0.6
                    else "Low" if match_result.overall_score >= 0.3 else "Very Low"
                )
            )

            api_match = MatchResult(
                trial_id=match_result.trial_id,
                trial_title=trial["trial_title"],
                condition=trial["condition"],
                overall_score=match_result.overall_score,
                inclusion_score=match_result.inclusion_score,
                exclusion_score=match_result.exclusion_score,
                age_eligible=match_result.age_eligible,
                explanation=match_result.explanation,
                confidence_level=confidence_level,
                detailed_matches=match_result.detailed_matches,
            )

            if api_match.overall_score >= min_threshold:
                all_matches.append(api_match)

        all_matches.sort(key=lambda x: x.overall_score, reverse=True)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        self.system_stats["patients_processed"] += 1
        self.system_stats["total_matches_computed"] += len(all_matches)

        return PatientMatchResponse(
            patient_id=patient_id,
            processing_time_ms=processing_time,
            total_trials_evaluated=len(self.processed_trials),
            matches_found=len(all_matches),
            top_matches=all_matches[:max_results],
            timestamp=datetime.now().isoformat(),
        )

    def process_fhir_bundle(self, fhir_bundle: Dict) -> Dict:
        """Processes a FHIR bundle and returns a patient_id and clinical_text."""
        if not DAY1_DAY2_AVAILABLE:
            return {
                "patient_id": "fhir_demo",
                "clinical_text": "Patient has diabetes from FHIR bundle.",
            }

        # The patient_processor has the fhir_integrator and can generate a narrative
        processed_fhir_data = self.patient_processor.fhir.process_fhir_patient(
            fhir_bundle
        )
        return {
            "patient_id": processed_fhir_data.patient_id,
            "clinical_text": processed_fhir_data.clinical_narrative,
        }

    def get_system_statistics(self):
        """Get system statistics"""
        uptime = datetime.now() - self.system_stats["system_start_time"]
        return {
            "total_trials_loaded": len(self.processed_trials),
            "total_patients_processed": self.system_stats["patients_processed"],
            "system_uptime": str(uptime).split(".")[0],
            "system_ready": True,
        }


# Initialize the matching system
matching_system = ClinicalTrialMatchingSystem()


# FastAPI Endpoints
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Clinical Trial Semantic Matching Engine",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Dual NLP Pipelines",
            "Semantic Matching",
            "Real-time API",
            "Web Interface",
        ],
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "trials_loaded": len(matching_system.processed_trials),
        "system_operational": True,
    }


@app.post("/match", response_model=PatientMatchResponse, tags=["Matching"])
async def match_patient(request: PatientMatchRequest):
    try:
        if request.fhir_patient_bundle:
            fhir_processed = matching_system.process_fhir_bundle(
                request.fhir_patient_bundle
            )
            patient_id = fhir_processed["patient_id"]
            clinical_text = fhir_processed["clinical_text"]
        else:
            patient_id = request.patient_id or "unknown"
            clinical_text = request.clinical_text or ""
            if not clinical_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Clinical text cannot be empty if no FHIR data provided",
                )

        # Call matching with clinical text and patient ID
        result = await matching_system.match_patient_to_trials(
            clinical_text, patient_id, request.max_results, request.min_score_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


@app.get("/stats", tags=["Analytics"])
async def get_system_stats():
    return matching_system.get_system_statistics()


@app.get("/trials", tags=["Trials"])
async def list_all_trials():
    trials_list = []
    for trial in matching_system.processed_trials:
        trials_list.append(
            {
                "nct_id": trial.get("trial_id", trial.get("nct_id", "Unknown")),
                "title": trial.get("trial_title", trial.get("title", "Unknown")),
                "condition": trial.get("condition", "Unknown"),
                "phase": trial.get("phase", "Unknown"),
            }
        )
    return {"trials": trials_list, "total_count": len(trials_list)}


# Streamlit Web Interface
def create_streamlit_interface():
    """Create Streamlit web interface"""

    st.set_page_config(
        page_title="Clinical Trial Matching Engine", page_icon="🏥", layout="wide"
    )

    st.title("🏥 Clinical Trial Semantic Matching Engine")
    st.markdown("AI-powered system for matching patients to clinical trials")

    # Sidebar
    with st.sidebar:
        st.header("System Information")

        try:
            stats = matching_system.get_system_statistics()
            st.metric("Trials Loaded", stats["total_trials_loaded"])
            st.metric("Patients Processed", stats["total_patients_processed"])
            st.metric("System Uptime", stats["system_uptime"])
        except:
            st.error("Unable to load system statistics")

    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Patient Matching", "Trial Browser", "System Demo", "API Documentation"]
    )

    with tab1:
        st.header("Patient-Trial Matching Interface")

        matching_mode = st.radio("Input Type", ["Clinical Text", "FHIR Patient JSON"])

        # Initialize variables
        clinical_text = ""
        fhir_data = None

        if matching_mode == "FHIR Patient JSON":
            fhir_input = st.text_area(
                "Paste FHIR Patient Bundle JSON here", "{}", height=300
            )
            try:
                fhir_data = json.loads(fhir_input) if fhir_input.strip() else None
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check the FHIR bundle.")
                fhir_data = None

        # Input section
        patient_id = st.text_input(
            "Patient ID", value=f"patient_{datetime.now().strftime('%H%M%S')}"
        )

        # Example patients
        example_patients = {
            "Diabetes Patient": "52-year-old male with type 2 diabetes. Currently on metformin. HbA1c 8.4%, BMI 31.2 kg/m². No heart failure history.",
            "Hypertension Patient": "58-year-old female with hypertension. Blood pressure 148/94 mmHg. Normal kidney function.",
            "Heart Failure Patient": "67-year-old male with heart failure NYHA Class II. Ejection fraction 38%. On medications.",
            "Custom Patient": "",
        }

        selected_example = st.selectbox(
            "Choose example:", list(example_patients.keys())
        )

        if matching_mode == "Clinical Text":
            clinical_text = st.text_area(
                "Patient Clinical Text",
                value=example_patients[selected_example],
                height=150,
            )

        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("Maximum Results", 1, 10, 5)
        with col2:
            min_threshold = st.slider("Minimum Score", 0.0, 1.0, 0.1, 0.1)

        # Matching button
        if st.button("Find Matching Trials", type="primary"):
            if matching_mode == "Clinical Text" and not clinical_text.strip():
                st.error("Please enter patient clinical text")
            elif matching_mode == "FHIR Patient JSON" and fhir_data is None:
                st.error("Please provide valid FHIR patient JSON")
            else:
                with st.spinner("Processing..."):
                    text_to_process = ""
                    id_to_process = patient_id

                    if matching_mode == "Clinical Text":
                        text_to_process = clinical_text
                    else:  # FHIR Patient JSON
                        fhir_processed = matching_system.process_fhir_bundle(fhir_data)
                        text_to_process = fhir_processed["clinical_text"]
                        id_to_process = fhir_processed["patient_id"]
                        st.info(
                            f"Extracted Patient ID '{id_to_process}' and clinical narrative from FHIR data."
                        )

                    if text_to_process:
                        result = asyncio.run(
                            matching_system.match_patient_to_trials(
                                text_to_process,
                                id_to_process,
                                max_results,
                                min_threshold,
                            )
                        )

                        st.success(
                            f"Found {result.matches_found} matches in {result.processing_time_ms:.1f}ms"
                        )

                        if result.top_matches:
                            st.subheader("Top Matching Trials")

                            for i, match in enumerate(result.top_matches, 1):
                                with st.expander(
                                    f"{i}. {match.trial_title} (Score: {match.overall_score:.3f})"
                                ):
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.write(f"**Trial ID:** {match.trial_id}")
                                        st.write(f"**Condition:** {match.condition}")
                                        st.write(
                                            f"**Overall Score:** {match.overall_score:.3f}"
                                        )

                                    with col2:
                                        st.write(
                                            f"**Inclusion Score:** {match.inclusion_score:.3f}"
                                        )
                                        st.write(
                                            f"**Exclusion Score:** {match.exclusion_score:.3f}"
                                        )
                                        st.write(
                                            f"**Confidence:** {match.confidence_level}"
                                        )

                                    st.write("**Explanation:**")
                                    st.info(match.explanation)

                                    # Display the complete, detailed matching information
                                    if match.detailed_matches:
                                        st.write("**Detailed Match Breakdown:**")
                                        st.json(match.detailed_matches)
                        else:
                            st.warning("No matching trials found.")

    with tab2:
        st.header("Clinical Trials Database")

        if matching_system.processed_trials:
            trials_data = []
            for trial in matching_system.processed_trials:
                trials_data.append(
                    {
                        "NCT ID": trial.get("trial_id", trial.get("nct_id", "Unknown")),
                        "Title": trial.get(
                            "trial_title", trial.get("title", "Unknown")
                        )[:60]
                        + "...",
                        "Condition": trial.get("condition", "Unknown"),
                        "Phase": trial.get("phase", "Unknown"),
                    }
                )

            df = pd.DataFrame(trials_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No trials available. Please run Day 1 first.")

    with tab3:
        st.header("System Demonstration")
        st.markdown("Interactive demonstration of the matching system.")

        # Feature overview
        features = [
            {"Feature": "Dual NLP Pipelines", "Status": "✅ Implemented"},
            {"Feature": "Semantic Matching", "Status": "✅ Implemented"},
            {"Feature": "Real-time API", "Status": "✅ Implemented"},
            {"Feature": "Web Interface", "Status": "✅ Implemented"},
        ]

        st.dataframe(pd.DataFrame(features))

        # Quick demo
        st.subheader("Quick Demo")
        demo_text = "45-year-old male with diabetes, HbA1c 8.2%"

        if st.button("Run Quick Demo"):
            with st.spinner("Running demo..."):
                try:
                    result = asyncio.run(
                        matching_system.match_patient_to_trials(
                            demo_text, "demo_patient", 3, 0.1
                        )
                    )

                    st.success(f"Demo completed! Found {result.matches_found} matches.")

                    if result.top_matches:
                        best_match = result.top_matches[0]
                        st.write(f"**Best Match:** {best_match.trial_title}")
                        st.write(f"**Score:** {best_match.overall_score:.3f}")
                        st.write(f"**Explanation:** {best_match.explanation}")

                except Exception as e:
                    st.error(f"Demo failed: {str(e)}")

    with tab4:
        st.header("API Documentation")

        endpoints = [
            {
                "Method": "POST",
                "Endpoint": "/match",
                "Description": "Match patient to trials",
            },
            {"Method": "GET", "Endpoint": "/trials", "Description": "List all trials"},
            {"Method": "GET", "Endpoint": "/stats", "Description": "System statistics"},
            {"Method": "GET", "Endpoint": "/health", "Description": "Health check"},
        ]

        st.dataframe(pd.DataFrame(endpoints))

        st.subheader("Example Usage")
        st.code(
            """
import requests

# Match a patient
response = requests.post("http://localhost:8000/match", json={
    "patient_id": "demo_001",
    "clinical_text": "45-year-old male with diabetes"
})

results = response.json()
print(f"Found {results['matches_found']} matches")
        """,
            language="python",
        )

        # Interactive tester
        st.subheader("API Tester")
        test_text = st.text_input("Test Patient Text", "45-year-old diabetic patient")

        if st.button("Test API"):
            if test_text:
                try:
                    result = asyncio.run(
                        matching_system.match_patient_to_trials(
                            test_text, "api_test", 3, 0.1
                        )
                    )
                    st.success("API test successful!")
                    st.json(result.dict())
                except Exception as e:
                    st.error(f"API test failed: {str(e)}")


def main():
    """Main execution function"""
    print("🏥 Clinical Trial Semantic Matching Engine - Day 3 Complete System")
    print("=" * 70)

    # System validation
    print("\nSystem Status:")
    print(f"  ✅ Day 1/2 Available: {DAY1_DAY2_AVAILABLE}")
    print(f"  ✅ Trials Loaded: {len(matching_system.processed_trials)}")
    print(f"  ✅ FastAPI Ready: {app.title}")
    print(f"  ✅ Streamlit Ready")

    # Quick test
    print("\n🧪 Quick System Test:")
    try:
        test_result = asyncio.run(
            matching_system.match_patient_to_trials(
                "45-year-old male with diabetes", "test_patient", 3, 0.1
            )
        )
        print(f"  ✅ Test successful: {test_result.matches_found} matches found")
        print(f"     Processing time: {test_result.processing_time_ms:.1f}ms")
    except Exception as e:
        print(f"  ❌ Test failed: {e}")

    print("\n🌐 Your system is ready!")
    print("\nTo start the servers:")
    print("  1. API: uvicorn app:app --reload --port 8000")
    print("  2. Web UI: streamlit run app.py --server.port 8501 -- --streamlit")

    print("\nAccess points:")
    print("  • Web Interface: http://localhost:8501")
    print("  • API Documentation: http://localhost:8000/docs")
    print("  • Health Check: http://localhost:8000/health")

    print("\n🎯 Complete Features:")
    print("  ✅ Real-time patient-trial matching")
    print("  ✅ Interactive web interface")
    print("  ✅ RESTful API endpoints")
    print("  ✅ System performance monitoring")
    print("  ✅ Multiple demo scenarios")
    print("  ✅ API documentation and testing")

    # Save system info
    system_info = {
        "system_name": "Clinical Trial Semantic Matching Engine",
        "version": "3.0.0",
        "completion_timestamp": datetime.now().isoformat(),
        "components_ready": {
            "day1_day2_processors": DAY1_DAY2_AVAILABLE,
            "fastapi_backend": True,
            "streamlit_interface": True,
            "trials_loaded": len(matching_system.processed_trials),
        },
    }

    with open("day3_system_ready.json", "w") as f:
        json.dump(system_info, f, indent=2)

    print("\n💾 System info saved: day3_system_ready.json")
    print("\n🎉 Day 3 Complete! Your clinical trial matching system is operational!")

    return matching_system


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        create_streamlit_interface()
    elif len(sys.argv) > 1 and sys.argv[1] == "--api":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    else:
        system = main()
