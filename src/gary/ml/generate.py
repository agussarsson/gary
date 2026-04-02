import json
from gary.api.schemas import ProgramJSON, ProgramGenerateRequest
import google.generativeai as genai
import os

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
except:
    print("No API key found.")

def build_program_generation_prompt(payload: ProgramGenerateRequest) -> str:
    prefs = ", ".join(payload.preferences) if payload.preferences else "none"
    equipment = payload.equipment if payload.equipment else "not specified"

    return f"""
You generate structured strength training programs.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations.
Do not include comments.

The JSON must match exactly this schema:
{{
  "split": string,
  "days_per_week": integer,
  "goal": string,
  "experience_level": string,
  "notes": string|null,
  "days": [
    {{
      "name": string,
      "focus": string,
      "exercises": [
        {{
          "name": string,
          "sets": integer,
          "reps": string,
          "load": number|null,
          "progression_rule": string
        }}
      ]
    }}
  ]
}}


"""
    

def validate_generated_program(raw_output: str) -> ProgramJSON:
    parsed = json.loads(raw_output)
    return ProgramJSON(**parsed)


  
def call_llm_for_program_generation(payload: ProgramGenerateRequest) -> ProgramJSON:
    prompt = build_program_generation_prompt(payload)

    raw_output = some_llm_client_call(prompt)

    raw_output = model.generate_content(prompt)

    return validate_generated_program(raw_output.text)