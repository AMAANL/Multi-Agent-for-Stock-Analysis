"""
Gemini client used by the stock analysis agents.

Gemini receives already-computed structured data and is used for
interpretation and summarization.

Supports:
    - Normal text responses
    - Structured JSON responses
    - Gemini response schemas
    - Automatic retry for temporary Gemini errors
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
    ):
        self.api_key = api_key or GEMINI_API_KEY

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.model = model

        self.client = genai.Client(
            api_key=self.api_key
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2500,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(
            **config_kwargs
        )

        # Retry temporary Gemini availability errors.
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )

                # Debug information. Useful while developing.
                if (
                    hasattr(response, "candidates")
                    and response.candidates
                ):
                    candidate = response.candidates[0]

                    finish_reason = getattr(
                        candidate,
                        "finish_reason",
                        None,
                    )

                    if finish_reason:
                        print(
                            f"[Gemini] Finish reason: "
                            f"{finish_reason}"
                        )

                    token_count = getattr(
                        candidate,
                        "token_count",
                        None,
                    )

                    if token_count:
                        print(
                            f"[Gemini] Token count: "
                            f"{token_count}"
                        )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text

            except Exception as e:
                error_text = str(e)

                # Gemini can temporarily return 503 when the
                # selected model is under heavy load.
                is_temporary = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text.upper()
                )

                # If this is not a temporary error, fail immediately.
                if not is_temporary:
                    raise RuntimeError(
                        f"Gemini API request failed: {e}"
                    ) from e

                # Last attempt failed.
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Gemini is temporarily unavailable "
                        "because the model is experiencing "
                        "high demand. Please try again later."
                    ) from e

                # Exponential backoff:
                # attempt 0 -> 2 seconds
                # attempt 1 -> 4 seconds
                wait_time = 2 ** (attempt + 1)

                print(
                    f"[Gemini] Temporary 503/UNAVAILABLE error. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Gemini API request failed after retries."
        )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        raw = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            response_schema=response_schema,
            **kwargs,
        )

        cleaned = raw.strip()

        # Remove markdown fences defensively.
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].strip().lower() in (
                "```json",
                "```",
            ):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)

        except json.JSONDecodeError as e:
            raise ValueError(
                "Gemini returned incomplete or invalid JSON.\n\n"
                f"Raw response:\n{raw}\n\n"
                "This usually means the model response was "
                "truncated. Increase max_tokens or check the "
                "Gemini finish reason."
            ) from e

        if not isinstance(result, dict):
            raise ValueError(
                "Gemini JSON response must be an object."
            )

        return result
