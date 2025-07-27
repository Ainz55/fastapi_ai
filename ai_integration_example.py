"""
Примеры интеграции различных ИИ сервисов в FastAPI приложение
"""

import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from fastapi import HTTPException


# Пример интеграции с OpenAI
class OpenAIIntegration:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    async def process_text(self, text: str, model: str = "gpt-3.5-turbo") -> \
    Dict[str, Any]:
        """Обработка текста через OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1000
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "input": text,
                        "output": result["choices"][0]["message"][
                            "content"],
                        "model": model,
                        "status": "success"
                    }
                else:
                    raise HTTPException(status_code=400,
                                        detail="OpenAI API error")


# Пример интеграции с Hugging Face
class HuggingFaceIntegration:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models"

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ тональности текста"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {"inputs": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/cardiffnlp/twitter-roberta-base-sentiment-latest",
                    headers=headers,
                    json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "text": text,
                        "sentiment": result[0][0]["label"],
                        "confidence": result[0][0]["score"],
                        "status": "success"
                    }
                else:
                    raise HTTPException(status_code=400,
                                        detail="Hugging Face API error")

    async def translate_text(self, text: str,
                             target_language: str = "en") -> Dict[str, Any]:
        """Перевод текста"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {"inputs": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/Helsinki-NLP/opus-mt-ru-en",
                    headers=headers,
                    json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "original": text,
                        "translated": result[0]["translation_text"],
                        "target_language": target_language,
                        "status": "success"
                    }
                else:
                    raise HTTPException(status_code=400,
                                        detail="Translation API error")


# Пример интеграции с локальной моделью
class LocalAIIntegration:
    def __init__(self):
        # Здесь можно инициализировать локальные модели
        # Например, используя transformers или другие библиотеки
        pass

    async def process_text_local(self, text: str) -> Dict[str, Any]:
        """Обработка текста локальной моделью"""
        # Пример простой обработки
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        return {
            "input": text,
            "output": f"Обработано локальной моделью. Слов: {word_count}, символов: {char_count}",
            "statistics": {
                "words": word_count,
                "characters": char_count,
                "average_word_length": char_count / word_count if word_count > 0 else 0
            },
            "status": "success"
        }

    async def classify_text(self, text: str) -> Dict[str, Any]:
        """Классификация текста"""
        # Простая классификация по ключевым словам
        text_lower = text.lower()

        categories = {
            "технологии": ["программирование", "компьютер", "технология",
                           "искусственный интеллект"],
            "наука": ["исследование", "эксперимент", "научный", "открытие"],
            "бизнес": ["компания", "прибыль", "рынок", "инвестиции"],
            "спорт": ["игра", "команда", "соревнование", "победа"]
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score

        best_category = max(scores, key=scores.get) if any(
            scores.values()) else "общее"

        return {
            "text": text,
            "category": best_category,
            "confidence_scores": scores,
            "status": "success"
        }


# Пример интеграции с Google Cloud AI
class GoogleCloudAIIntegration:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://language.googleapis.com/v1"

    async def analyze_entities(self, text: str) -> Dict[str, Any]:
        """Анализ именованных сущностей"""
        headers = {"Content-Type": "application/json"}

        data = {
            "document": {
                "type": "PLAIN_TEXT",
                "content": text
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/documents:analyzeEntities?key={self.api_key}",
                    headers=headers,
                    json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    entities = result.get("entities", [])
                    return {
                        "text": text,
                        "entities": [{"name": e["name"], "type": e["type"]}
                                     for e in entities],
                        "entity_count": len(entities),
                        "status": "success"
                    }
                else:
                    raise HTTPException(status_code=400,
                                        detail="Google Cloud AI error")


# Фабрика для создания ИИ интеграций
class AIIntegrationFactory:
    @staticmethod
    def create_integration(integration_type: str,
                           api_key: Optional[str] = None):
        """Создание экземпляра ИИ интеграции по типу"""
        if integration_type == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI integration")
            return OpenAIIntegration(api_key)
        elif integration_type == "huggingface":
            if not api_key:
                raise ValueError(
                    "API key required for Hugging Face integration")
            return HuggingFaceIntegration(api_key)
        elif integration_type == "local":
            return LocalAIIntegration()
        elif integration_type == "google":
            if not api_key:
                raise ValueError(
                    "API key required for Google Cloud AI integration")
            return GoogleCloudAIIntegration(api_key)
        else:
            raise ValueError(
                f"Unknown integration type: {integration_type}")


# Пример использования в FastAPI
"""
# В main.py добавьте:

from ai_integration_example import AIIntegrationFactory
import os

# Инициализация ИИ интеграций
ai_factory = AIIntegrationFactory()
openai_ai = ai_factory.create_integration("openai", os.getenv("OPENAI_API_KEY"))
huggingface_ai = ai_factory.create_integration("huggingface", os.getenv("HF_API_KEY"))
local_ai = ai_factory.create_integration("local")

@app.post("/api/ai/process-text")
async def process_text_with_ai(
    text: str = Form(...),
    ai_type: str = Form("local")
):
    try:
        if ai_type == "openai":
            result = await openai_ai.process_text(text)
        elif ai_type == "huggingface":
            result = await huggingface_ai.analyze_sentiment(text)
        elif ai_type == "local":
            result = await local_ai.process_text_local(text)
        else:
            raise HTTPException(status_code=400, detail="Unknown AI type")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/classify")
async def classify_text(text: str = Form(...)):
    try:
        result = await local_ai.classify_text(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/translate")
async def translate_text(
    text: str = Form(...),
    target_lang: str = Form("en")
):
    try:
        result = await huggingface_ai.translate_text(text, target_lang)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

if __name__ == "__main__":
    # Пример тестирования
    async def test_integrations():
        local_ai = LocalAIIntegration()

        # Тест обработки текста
        result = await local_ai.process_text_local(
            "Привет, это тестовый текст для обработки ИИ!")
        print("Обработка текста:",
              json.dumps(result, ensure_ascii=False, indent=2))

        # Тест классификации
        result = await local_ai.classify_text(
            "Новое исследование в области искусственного интеллекта")
        print("Классификация:",
              json.dumps(result, ensure_ascii=False, indent=2))


    asyncio.run(test_integrations())
