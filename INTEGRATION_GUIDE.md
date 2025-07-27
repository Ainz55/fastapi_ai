# Руководство по интеграции ИИ в FastAPI проект

## 🎯 Обзор

Это руководство поможет вам интегрировать различные ИИ сервисы и модели в ваш FastAPI веб-сайт.

## 🚀 Быстрый старт

### 1. Базовая интеграция

Самый простой способ - отредактировать существующие функции в `main.py`:

```python
@app.post("/api/process-text")
async def process_text(text: str = Form(...)):
    # Замените эту строку на вашу ИИ логику:
    # result = your_ai_model.process(text)
    
    # Пример простой обработки:
    processed_text = f"Обработанный текст: {text.upper()}"
    
    return {
        "input": text,
        "output": processed_text,
        "status": "success"
    }
```

### 2. Использование готовых интеграций

Используйте файл `ai_integration_example.py` для интеграции популярных ИИ сервисов.

## 🔧 Интеграция с OpenAI

### Установка
```bash
pip install openai
```

### Настройка
1. Получите API ключ на [platform.openai.com](https://platform.openai.com)
2. Добавьте ключ в переменные окружения:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Интеграция в main.py
```python
import openai
import os

# Инициализация
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/api/openai/chat")
async def chat_with_gpt(text: str = Form(...)):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=1000
        )
        
        return {
            "input": text,
            "output": response.choices[0].message.content,
            "model": "gpt-3.5-turbo",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🤗 Интеграция с Hugging Face

### Установка
```bash
pip install transformers torch
```

### Локальное использование
```python
from transformers import pipeline

# Инициализация модели
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@app.post("/api/hf/sentiment")
async def analyze_sentiment(text: str = Form(...)):
    try:
        result = classifier(text)
        return {
            "text": text,
            "sentiment": result[0]["label"],
            "confidence": result[0]["score"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Использование Hugging Face API
```python
import aiohttp
import os

async def analyze_sentiment_api(text: str):
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    data = {"inputs": text}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
            headers=headers,
            json=data
        ) as response:
            return await response.json()
```

## 🧠 Интеграция локальных моделей

### Использование scikit-learn
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Загрузка обученной модели
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

vectorizer = TfidfVectorizer()

@app.post("/api/local/classify")
async def classify_text(text: str = Form(...)):
    # Векторизация текста
    text_vectorized = vectorizer.transform([text])
    
    # Предсказание
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized).max()
    
    return {
        "text": text,
        "prediction": prediction,
        "confidence": probability,
        "status": "success"
    }
```

### Использование spaCy
```python
import spacy

# Загрузка модели
nlp = spacy.load("en_core_web_sm")

@app.post("/api/spacy/analyze")
async def analyze_text(text: str = Form(...)):
    doc = nlp(text)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    tokens = [token.text for token in doc]
    
    return {
        "text": text,
        "entities": entities,
        "tokens": tokens,
        "status": "success"
    }
```

## 📊 Интеграция с Google Cloud AI

### Установка
```bash
pip install google-cloud-language
```

### Настройка
1. Создайте проект в Google Cloud Console
2. Включите Natural Language API
3. Создайте сервисный аккаунт и скачайте JSON ключ
4. Установите переменную окружения:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

### Интеграция
```python
from google.cloud import language_v1

client = language_v1.LanguageServiceClient()

@app.post("/api/google/analyze")
async def analyze_text_google(text: str = Form(...)):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    
    # Анализ тональности
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    
    # Анализ сущностей
    entities = client.analyze_entities(request={'document': document}).entities
    
    return {
        "text": text,
        "sentiment": {
            "score": sentiment.score,
            "magnitude": sentiment.magnitude
        },
        "entities": [{"name": e.name, "type": e.type_.name} for e in entities],
        "status": "success"
    }
```

## 🎨 Обновление фронтенда

### Добавление новых функций в ai_demo.html

```html
<!-- Добавьте новую секцию для ИИ функций -->
<div class="row mb-4">
    <div class="col-lg-6">
        <div class="card border-0 shadow-sm">
            <div class="card-header bg-info text-white">
                <h5 class="mb-0">
                    <i class="fas fa-brain me-2"></i>Анализ тональности
                </h5>
            </div>
            <div class="card-body">
                <form id="sentimentForm">
                    <div class="mb-3">
                        <label for="sentimentText" class="form-label">Введите текст для анализа:</label>
                        <textarea class="form-control" id="sentimentText" rows="3"></textarea>
                    </div>
                    <button type="submit" class="btn btn-info">
                        <i class="fas fa-chart-line me-2"></i>Анализировать
                    </button>
                </form>
            </div>
        </div>
    </div>
    <div class="col-lg-6">
        <div class="card border-0 shadow-sm">
            <div class="card-header bg-warning text-dark">
                <h5 class="mb-0">
                    <i class="fas fa-lightbulb me-2"></i>Результат анализа
                </h5>
            </div>
            <div class="card-body">
                <div id="sentimentResult">
                    <p class="text-muted text-center">Результат появится здесь</p>
                </div>
            </div>
        </div>
    </div>
</div>
```

### JavaScript для новых функций

```javascript
// Добавьте в ai_demo.html в секцию scripts

// Анализ тональности
document.getElementById('sentimentForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const text = document.getElementById('sentimentText').value;
    
    if (!text.trim()) {
        alert('Пожалуйста, введите текст для анализа');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('text', text);
        
        const response = await fetch('/api/hf/sentiment', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        document.getElementById('sentimentResult').innerHTML = `
            <div class="alert alert-success">
                <h6>Результат анализа:</h6>
                <p><strong>Тональность:</strong> ${result.sentiment}</p>
                <p><strong>Уверенность:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
            </div>
        `;
    } catch (error) {
        document.getElementById('sentimentResult').innerHTML = `
            <div class="alert alert-danger">
                <h6>Ошибка:</h6>
                <p>Произошла ошибка при анализе</p>
            </div>
        `;
    } finally {
        hideLoading();
    }
});
```

## 🔒 Безопасность

### Валидация входных данных
```python
from pydantic import BaseModel, validator

class TextInput(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Текст не может быть пустым')
        if len(v) > 10000:
            raise ValueError('Текст слишком длинный')
        return v.strip()

@app.post("/api/secure/process")
async def secure_process_text(input_data: TextInput):
    # Ваша ИИ логика здесь
    return {"result": "processed"}
```

### Ограничение размера файлов
```python
from fastapi import UploadFile, File

@app.post("/api/upload-secure")
async def upload_secure_file(file: UploadFile = File(...)):
    # Проверка размера файла (10MB)
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой")
    
    # Проверка типа файла
    allowed_types = [".txt", ".pdf", ".doc", ".docx"]
    if not any(file.filename.endswith(ext) for ext in allowed_types):
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла")
```

## 📈 Мониторинг и логирование

```python
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/api/ai/process")
async def process_with_logging(text: str = Form(...)):
    start_time = datetime.now()
    
    try:
        # Ваша ИИ логика
        result = await your_ai_function(text)
        
        # Логирование успешного запроса
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"AI processing completed in {processing_time}s for text: {text[:50]}...")
        
        return result
    except Exception as e:
        # Логирование ошибки
        logger.error(f"AI processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## 🚀 Развертывание

### Docker с ИИ моделями
```dockerfile
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p uploads models

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Переменные окружения
Создайте файл `.env`:
```env
OPENAI_API_KEY=your-openai-key
HF_API_KEY=your-huggingface-key
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
MODEL_PATH=/app/models
MAX_FILE_SIZE=10485760
```

## 🎯 Примеры использования

### Чат-бот с памятью
```python
from collections import defaultdict

# Простое хранилище для контекста
conversation_history = defaultdict(list)

@app.post("/api/chat")
async def chat_with_memory(
    message: str = Form(...),
    user_id: str = Form(...)
):
    # Добавление сообщения в историю
    conversation_history[user_id].append({"role": "user", "content": message})
    
    # Ограничение истории (последние 10 сообщений)
    if len(conversation_history[user_id]) > 10:
        conversation_history[user_id] = conversation_history[user_id][-10:]
    
    # Формирование контекста для ИИ
    context = "\n".join([msg["content"] for msg in conversation_history[user_id]])
    
    # Обработка через ИИ
    response = await your_ai_model.process(context)
    
    # Сохранение ответа
    conversation_history[user_id].append({"role": "assistant", "content": response})
    
    return {"response": response, "history_length": len(conversation_history[user_id])}
```

### Анализ документов
```python
import PyPDF2
import docx

@app.post("/api/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    content = ""
    
    if file.filename.endswith('.pdf'):
        # Чтение PDF
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            content += page.extract_text()
    elif file.filename.endswith('.docx'):
        # Чтение DOCX
        doc = docx.Document(file.file)
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
    
    # Анализ через ИИ
    analysis = await your_ai_model.analyze_document(content)
    
    return {
        "filename": file.filename,
        "content_length": len(content),
        "analysis": analysis
    }
```

## 📚 Дополнительные ресурсы

- [FastAPI документация](https://fastapi.tiangolo.com/)
- [Hugging Face документация](https://huggingface.co/docs)

---

**Удачной интеграции! 🚀** 