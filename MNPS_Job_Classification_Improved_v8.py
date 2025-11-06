# MNPS Job Classification - Improved Two-Pass Approach v8.0
# Implements recommendations for better accuracy through enhanced self-consistency
# and minimal post-processing

import os
import json
import datetime as dt
import pandas as pd
import numpy as np
import re
import time
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ===== 1. SETUP AND DATA LOADING =====

# Initialize paths
RUN_ROOT = Path('/content')
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = f"RUN_{timestamp}_v8"
OUTPUTS_DIR = Path(f"/content/outputs_{run_folder}")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Outputs directory: {OUTPUTS_DIR}")

# Load data files
BATCH_INPUT_CSV = RUN_ROOT / "Sample JDs.csv"
MNPS_ROLES_CSV = RUN_ROOT / "MNPS Roles.csv"
MNPS_KSACS_CSV = RUN_ROOT / "MNPS KSACs.csv"
COMPETENCY_EXTENDED_CSV = RUN_ROOT / "Competency Extended Descriptions.csv"
KORN_FERRY_CSV = RUN_ROOT / "Korn_Ferry Lominger 38 Competencies.csv"

# Load all dataframes
df = pd.read_csv(BATCH_INPUT_CSV)
roles_df = pd.read_csv(MNPS_ROLES_CSV)
ksacs_df = pd.read_csv(MNPS_KSACS_CSV)
competency_df = pd.read_csv(COMPETENCY_EXTENDED_CSV)
korn_ferry_df = pd.read_csv(KORN_FERRY_CSV)

print(f"✅ Loaded {len(df)} job descriptions")

# ===== 2. VALID ROLES AND NORMALIZATION =====

# Extract valid roles from MNPS Roles file
VALID_ROLES = sorted(roles_df['Role'].dropna().unique().tolist())
print(f"✅ Found {len(VALID_ROLES)} valid MNPS roles")

# Define role categories
EXECUTIVE_ROLES = ['Coordinator', 'Principal', 'Director', 'Manager']
NO_SUBGROUP_ROLES = ['Teacher', 'Assistant Principal', 'Principal']

def normalize_minor(minor_str):
    """Normalize minor role to standard format."""
    if not minor_str or minor_str == 'nan' or pd.isna(minor_str):
        return ''
    minor_str = str(minor_str).strip()
    
    # Handle numeric conversions
    conversions = {
        '1': 'I', 'i': 'I', 'I': 'I',
        '2': 'II', 'ii': 'II', 'II': 'II',
        '3': 'III', 'iii': 'III', 'III': 'III',
        'Lead': 'Lead', 'lead': 'Lead', 'LEAD': 'Lead'
    }
    
    return conversions.get(minor_str, minor_str)

# ===== 3. BUILD KSACS TEXT =====

def build_ksacs_text():
    """Build comprehensive KSACs text from all MNPS resources."""
    ksacs_text = "MNPS Knowledge, Skills, Abilities, and Competencies (KSACs):\n"
    
    # Add role-specific KSACs
    for _, row in ksacs_df.iterrows():
        role = row.get('Role', '')
        ksacs = row.get('KSACs', '')
        if role and ksacs:
            ksacs_text += f"**{role}**:\n{ksacs}\n"
    
    # Add competency descriptions
    ksacs_text += "\n**Competency Extended Descriptions**:\n"
    for _, row in competency_df.iterrows():
        competency = row.get('Competency', '')
        description = row.get('Extended Description', '')
        if competency and description:
            ksacs_text += f"- {competency}: {description}\n"
    
    # Add Korn Ferry competencies
    ksacs_text += "\n**Korn Ferry Lominger 38 Competencies**:\n"
    for _, row in korn_ferry_df.iterrows():
        competency = row.get('Competency', '')
        definition = row.get('Description', '')
        if competency and definition:
            ksacs_text += f"- {competency}: {definition}\n"
    
    return ksacs_text

KSACS_TEXT = build_ksacs_text()
print(f"✅ Built KSACs text ({len(KSACS_TEXT)} characters)")

# ===== 4. PROMPTS =====

# Initial classification prompt
ZERO_SHOT_PROMPT = """Objective: Classify this job based on its functions and requirements, NOT its title.

Process:
- Analyze the job attributes: Education, Work Experience, Essential Functions, KSAs
- Match to MNPS role classifications based on actual work performed
- Provide clear justification based on job attributes and MNPS standards

IMPORTANT ROLE DISTINCTIONS:
- **Technician**: Hands-on technical work, equipment maintenance, repair
- **Specialist**: Specialized domain knowledge (avoid overuse - prefer specific roles)
- **Analyst**: Data analysis, research, evaluation, reporting
- **Coordinator**: Program coordination, organization, facilitation, liaison work
- **Coach**: Instructional support, mentoring, professional development, co-teaching
- **Manager**: Strategic planning, policy, budget oversight, supervision
- **Supervisor**: Primarily people management, no degree required
- **Assistant**: Supporting role in specialized function

MINOR SUB-GROUP GUIDELINES:
- **I**: Entry-level, basic complexity
- **II**: Intermediate complexity and responsibility
- **III**: Advanced KSACs, senior-level expertise
- **Lead**: Leads teams/projects (rare for executive roles)
- Some roles typically have NO minor sub-group (leave blank)

Output Requirements:
Return JSON with:
- new_job_title: Descriptive title with role and level
- major_role_group: From approved MNPS roles
- minor_sub_group: I, II, III, Lead, or blank
- grouping_justification: Detailed explanation matching your classification"""

# Enhanced self-consistency prompt
SELF_CONSISTENCY_PROMPT = """TASK: Review your classification and ensure PERFECT alignment.

CRITICAL CHECKS:
1. Does major_role_group match the justification?
   - If justification describes "coordination/organizing" → major_role_group MUST be "Coordinator"
   - If justification describes "instructional support/mentoring" → major_role_group MUST be "Coach"
   - If justification describes "strategic planning/supervision" → major_role_group MUST be "Manager"
   - If justification describes "data analysis/research" → major_role_group MUST be "Analyst"
   - If justification describes "hands-on technical work" → major_role_group MUST be "Technician"

2. Is minor_sub_group appropriate?
   - Teacher: Usually NO sub-group (blank) unless explicitly "Lead Teacher"
   - Assistant Principal: Usually NO sub-group (blank)
   - Principal: Usually NO sub-group (blank)
   - Entry-level positions: "I"
   - Intermediate complexity: "II"
   - Senior/advanced: "III"

3. Common errors to fix:
   - Coordinator vs Coach confusion (Coordinators organize programs; Coaches teach/mentor)
   - Manager vs Coordinator (Managers have budget/strategic duties; Coordinators facilitate)
   - Overuse of "Specialist" (prefer specific roles like Analyst, Technician)

INSTRUCTIONS:
- If ANY mismatch exists, FIX the classification to match the justification
- Keep the justification unchanged
- Return corrected JSON with same structure"""

# ===== 5. OPENAI SETUP =====

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_ID = "gpt-4o-2024-11-20"
print(f"✅ Using model: {MODEL_ID}")

def call_llm_json_with_retry(prompt: str, model: str = MODEL_ID, max_retries: int = 3) -> dict:
    """Call OpenAI API with retry logic for rate limiting."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ Rate limit, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
            raise e
    raise Exception("Max retries reached")

# ===== 6. MINIMAL POST-PROCESSING =====

def minimal_post_processing(major_role, minor_role, job_text, justification):
    """Apply only essential post-processing rules."""
    
    # 1. Normalize minor role format
    minor_role = normalize_minor(minor_role)
    
    # 2. Handle roles that typically don't have sub-groups
    if major_role in NO_SUBGROUP_ROLES:
        # Only keep 'Lead' designation if explicitly mentioned
        if minor_role and minor_role != 'Lead':
            if 'lead' not in job_text.lower() and 'lead' not in justification.lower():
                minor_role = ''
    
    # 3. Fix ONLY if there's clear mismatch between justification and role
    justification_lower = justification.lower()
    
    # Check for clear mismatches
    if major_role == 'Coach' and 'coach' not in justification_lower:
        if 'coordinat' in justification_lower:
            major_role = 'Coordinator'
    elif major_role == 'Coordinator' and 'coordinat' not in justification_lower:
        if 'coach' in justification_lower or 'instruct' in justification_lower:
            major_role = 'Coach'
    
    # 4. Executive roles rarely have "Lead"
    if major_role in EXECUTIVE_ROLES and minor_role == 'Lead':
        # Downgrade to III for senior executives, II otherwise
        if major_role in ['Director', 'Principal']:
            minor_role = 'III'
        else:
            minor_role = 'II'
    
    return major_role, minor_role

# ===== 7. VALIDATION =====

def validate_classification(major_role, minor_role, justification):
    """Validate that classification matches justification."""
    issues = []
    justification_lower = justification.lower()
    
    # Role keywords to check
    role_keywords = {
        'Coordinator': ['coordinat', 'organiz', 'facilitat', 'liaison'],
        'Coach': ['coach', 'instruct', 'mentor', 'professional development'],
        'Manager': ['manag', 'strategic', 'supervis', 'budget', 'policy'],
        'Analyst': ['analy', 'data', 'research', 'evaluat', 'assess'],
        'Technician': ['technical', 'hands-on', 'repair', 'maintain', 'equipment'],
        'Teacher': ['teach', 'instruct', 'lesson', 'classroom', 'student'],
        'Assistant': ['assist', 'support', 'help'],
        'Specialist': ['specializ', 'expert'],
        'Therapist': ['therap', 'treatment', 'intervention'],
        'Principal': ['principal', 'school leader'],
        'Director': ['director', 'oversee', 'department']
    }
    
    # Check if role appears in justification
    if major_role in role_keywords:
        found_keyword = False
        for keyword in role_keywords[major_role]:
            if keyword in justification_lower:
                found_keyword = True
                break
        
        if not found_keyword:
            issues.append(f"Role '{major_role}' not aligned with justification")
    
    return issues

# ===== 8. MAIN PROCESSING FUNCTION =====

def process_job_description_improved(row_idx: int, row: pd.Series) -> dict:
    """Improved two-pass processing with better self-consistency."""
    
    # Get original job title for reference
    job_title_original = row.get('Job Description Name', '')
    
    # Build job text (ignoring title)
    job_text = f"""Position Summary: {row.get('Position Summary', '')}
Essential Functions: {row.get('Essential Functions', '')}
Work Experience: {row.get('Work Experience', '')}
Education: {row.get('Education', '')}
Licenses and Certifications: {row.get('Licenses and Certifications', '')}
Knowledge, Skills and Abilities: {row.get('Knowledge, Skills and Abilities', '')}"""
    
    # === PASS 1: Initial Classification ===
    pass1_prompt = f"""{ZERO_SHOT_PROMPT}

Available MNPS Roles: {', '.join(VALID_ROLES)}

{KSACS_TEXT}

Job Description to Classify:
{job_text}

IMPORTANT: Base classification solely on job attributes, not the title.

Return JSON with: new_job_title, major_role_group, minor_sub_group, grouping_justification"""
    
    try:
        # Get initial classification
        pass1_response = call_llm_json_with_retry(pass1_prompt)
        
        raw_major = pass1_response.get('major_role_group', 'Other')
        raw_minor = pass1_response.get('minor_sub_group', 'I')
        justification = pass1_response.get('grouping_justification', 'No justification')
        
        # === PASS 2: Self-Consistency Check ===
        pass2_prompt = f"""{SELF_CONSISTENCY_PROMPT}

Your Previous Output:
{{
  "major_role_group": "{raw_major}",
  "minor_sub_group": "{raw_minor}",
  "grouping_justification": "{justification}"
}}

Review for mismatches and return corrected JSON."""
        
        pass2_response = call_llm_json_with_retry(pass2_prompt)
        
        # Extract corrected values
        final_major = pass2_response.get('major_role_group', raw_major)
        final_minor = pass2_response.get('minor_sub_group', raw_minor)
        final_justification = pass2_response.get('grouping_justification', justification)
        
        # Apply MINIMAL post-processing
        final_major, final_minor = minimal_post_processing(
            final_major, final_minor, job_text, final_justification
        )
        
        # Validate alignment
        validation_issues = validate_classification(final_major, final_minor, final_justification)
        
        # Construct title
        if final_minor:
            new_job_title = f"{final_major} {final_minor}"
        else:
            new_job_title = final_major
        
        return {
            'source_row_index': row_idx,
            'job_title_original': job_title_original,
            'new_job_title': new_job_title,
            'major_role_group': final_major,
            'minor_sub_group': final_minor,
            'grouping_justification': final_justification,
            'validation_issues': '; '.join(validation_issues) if validation_issues else 'None',
            'model_used': MODEL_ID
        }
        
    except Exception as e:
        print(f"❌ Error processing row {row_idx}: {e}")
        return {
            'source_row_index': row_idx,
            'job_title_original': job_title_original,
            'new_job_title': 'Error',
            'major_role_group': 'Other',
            'minor_sub_group': 'I',
            'grouping_justification': f'Error: {str(e)}',
            'validation_issues': 'Processing error',
            'model_used': MODEL_ID
        }

# ===== 9. BATCH PROCESSING =====

print("\n🚀 Starting improved two-pass batch processing...")
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing jobs"):
    result = process_job_description_improved(idx, row)
    results.append(result)
    
    # Rate limiting protection
    time.sleep(0.3)
    
    # Print validation issues if any
    if result['validation_issues'] != 'None':
        print(f"⚠️ Row {idx}: {result['validation_issues']}")

# ===== 10. SAVE RESULTS =====

results_df = pd.DataFrame(results)
output_path = OUTPUTS_DIR / "Job_Classifications_v8_Improved.csv"
results_df.to_csv(output_path, index=False)
print(f"\n✅ Saved results to: {output_path}")

# ===== 11. SUMMARY STATISTICS =====

print("\n📊 Classification Summary:")
print("-" * 40)
print(f"Total jobs processed: {len(results_df)}")
print(f"Unique major roles: {results_df['major_role_group'].nunique()}")
print(f"Unique minor roles: {results_df['minor_sub_group'].nunique()}")

# Count validation issues
issues_df = results_df[results_df['validation_issues'] != 'None']
print(f"Classifications with validation issues: {len(issues_df)}")

# Major role distribution
print("\nTop 10 Major Roles:")
print(results_df['major_role_group'].value_counts().head(10))

# Minor role distribution
print("\nMinor Role Distribution:")
print(results_df['minor_sub_group'].value_counts())

# Check specific problem areas
print("\nProblem Area Analysis:")
coordinator_coach_issues = results_df[
    (results_df['major_role_group'].isin(['Coordinator', 'Coach'])) &
    (results_df['validation_issues'] != 'None')
]
print(f"Coordinator/Coach validation issues: {len(coordinator_coach_issues)}")

# Executive roles with "Lead"
exec_lead = results_df[
    (results_df['major_role_group'].isin(EXECUTIVE_ROLES)) &
    (results_df['minor_sub_group'] == 'Lead')
]
print(f"Executive roles with 'Lead' designation: {len(exec_lead)}")

# Roles without subgroups
no_subgroup = results_df[
    (results_df['major_role_group'].isin(NO_SUBGROUP_ROLES)) &
    (results_df['minor_sub_group'] == '')
]
print(f"Teacher/Principal roles without subgroups: {len(no_subgroup)}")

# Save summary statistics
summary_stats = {
    'total_jobs': len(results_df),
    'unique_major_roles': results_df['major_role_group'].nunique(),
    'unique_minor_roles': results_df['minor_sub_group'].nunique(),
    'validation_issues': len(issues_df),
    'coordinator_coach_issues': len(coordinator_coach_issues),
    'executive_lead_count': len(exec_lead),
    'no_subgroup_correct': len(no_subgroup)
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(OUTPUTS_DIR / "summary_stats_v8.csv", index=False)
print(f"\n✅ Summary statistics saved")

print("\n✨ Processing complete! Review the outputs for detailed results.")
