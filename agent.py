"""
agent.py
--------
Purpose:
    Core orchestration agent for the AI Data Analyst system.
    Interfaces with Google Gemini API using the official google-genai SDK.
    Translates natural language questions into executable Pandas code based on dataset schema.
"""

import os
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt import build_system_prompt, build_user_prompt
from validator import clean_code

# Load environment variables from .env file
load_dotenv()


class DataAnalystAgent:
    """
    Orchestration agent for converting natural language questions into Pandas code.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the agent with Gemini API configuration.

        Args:
            api_key (Optional[str]): Gemini API key. If None, loads from GEMINI_API_KEY env var.
            model_name (str): Target Gemini model identifier (default: gemini-2.5-flash).

        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key or self.api_key == "your_google_gemini_api_key_here":
            raise ValueError(
                "Gemini API Key is missing. Please add your GEMINI_API_KEY to the .env file."
            )
            
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def generate_code(self, df: pd.DataFrame, user_question: str) -> str:
        """
        Calls Google Gemini API to translate a user question into pandas code.

        Args:
            df (pd.DataFrame): Target dataset DataFrame.
            user_question (str): Plain English question.

        Returns:
            str: Generated python pandas code.

        Raises:
            ValueError: If user_question is empty or API returns an empty response.
            RuntimeError: If Gemini API invocation fails.
        """
        if not user_question or not user_question.strip():
            raise ValueError("User question cannot be empty.")

        system_prompt = build_system_prompt(df)
        user_prompt = build_user_prompt(user_question)

        try:
            # Invoke Gemini model using google-genai SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1
                )
            )

            if not response.text or not response.text.strip():
                raise ValueError("Gemini API returned an empty response. Please try rephrasing your question.")

            return clean_code(response.text)

        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise RuntimeError(f"Gemini API Request Failed: {str(e)}")
