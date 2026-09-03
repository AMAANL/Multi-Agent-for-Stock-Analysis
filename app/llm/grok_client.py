"""
Thin wrapper around the xAI Grok Chat Completions API.

Grok receives already-computed structured data and is used
for interpretation/summarization rather than calculations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "gemini-2.5-flash")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")


class GrokClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROK_MODEL,
        base_url: str = GROK_BASE_URL,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or GROK_API_KEY

        if not self.api_key:
            raise ValueError(
                "GROK_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        if json_mode:
            payload["response_format"] = {
                "type": "json_object"
            }

        url = f"{self.base_url}/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                )

                resp.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Grok API request failed: "
                    f"{resp.status_code} "
                    f"{resp.reason_phrase} - "
                    f"{resp.text}"
                ) from e

        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected Grok API response: {data}"
            ) from e

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:

        raw = self.chat(
            system_prompt,
            user_prompt,
            json_mode=True,
            **kwargs,
        )

        cleaned = raw.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].lower().startswith("```json"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Grok did not return valid JSON: {raw[:500]}"
            ) from e