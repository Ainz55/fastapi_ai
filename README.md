# AI Project Website

Современный веб-сайт на FastAPI для демонстрации и интеграции ИИ проектов.

## 🚀 Возможности

- **Современный веб-интерфейс** с адаптивным дизайном
- **Обработка текста** через ИИ API
- **Загрузка и обработка файлов**
- **RESTful API** для интеграции с ИИ моделями
- **Интерактивная демонстрация** возможностей
- **Красивый UI/UX** с анимациями и эффектами

## 🛠 Технологический стек

- **Backend**: FastAPI, Python 3.8+
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Сервер**: Uvicorn
- **Шаблоны**: Jinja2
- **Стили**: Custom CSS с анимациями

## 📦 Установка

1. **Клонируйте репозиторий:**
```bash
git clone <your-repo-url>
cd fastapi_for_sayt
```

2. **Создайте виртуальное окружение:**
```bash
python -m venv venv
```

3. **Активируйте виртуальное окружение:**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

## 🚀 Запуск

1. **Запустите сервер разработки:**
```bash
python main.py
```

2. **Или используйте uvicorn напрямую:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Откройте браузер и перейдите по адресу:**
```
http://localhost:8000
```

## 📁 Структура проекта

```
fastapi_for_sayt/
├── main.py              # Основной файл FastAPI приложения
├── requirements.txt     # Зависимости Python
├── README.md           # Документация
├── templates/          # HTML шаблоны
│   ├── base.html       # Базовый шаблон
│   ├── index.html      # Главная страница
│   ├── ai_demo.html    # Страница демонстрации ИИ
│   └── about.html      # Страница о проекте
├── static/             # Статические файлы
│   ├── css/
│   │   └── style.css   # Пользовательские стили
│   └── js/
│       └── main.js     # JavaScript функциональность
└── uploads/            # Директория для загруженных файлов
```

## 🔧 API Endpoints

### Обработка текста
- **POST** `/api/process-text`
- **Параметры**: `text` (form data)
- **Возвращает**: JSON с обработанным текстом

### Загрузка файлов
- **POST** `/api/upload-file`
- **Параметры**: `file` (multipart/form-data)
- **Возвращает**: JSON с информацией о файле

## 🎨 Интеграция ИИ

Для интеграции вашего ИИ проекта:

1. **Отредактируйте функции в `main.py`:**
```python
@app.post("/api/process-text")
async def process_text(text: str = Form(...)):
    # Здесь добавьте вашу логику ИИ
    # Например:
    # result = your_ai_model.process(text)
    return {
        "input": text,
        "output": "Ваш результат ИИ",
        "status": "success"
    }
```

2. **Добавьте новые API endpoints** для специфичных функций вашего ИИ

3. **Обновите фронтенд** в `templates/ai_demo.html` для новых возможностей

## 🎯 Примеры интеграции

### Интеграция с OpenAI
```python
import openai

@app.post("/api/process-text")
async def process_text(text: str = Form(...)):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
    return {
        "input": text,
        "output": response.choices[0].message.content,
        "status": "success"
    }
```

### Интеграция с Hugging Face
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

@app.post("/api/analyze-sentiment")
async def analyze_sentiment(text: str = Form(...)):
    result = classifier(text)
    return {
        "text": text,
        "sentiment": result[0]["label"],
        "confidence": result[0]["score"]
    }
```

## 🎨 Кастомизация

### Изменение стилей
- Отредактируйте `static/css/style.css`
- Добавьте новые CSS классы
- Измените цветовую схему

### Добавление новых страниц
1. Создайте новый HTML файл в `templates/`
2. Добавьте роут в `main.py`
3. Обновите навигацию в `templates/base.html`

### Расширение API
1. Добавьте новые endpoints в `main.py`
2. Обновите фронтенд для использования новых API
3. Добавьте валидацию данных с Pydantic

## 🔒 Безопасность

- Валидация входных данных
- Ограничение размера файлов
- Проверка типов файлов
- Защита от CSRF атак

## 📊 Мониторинг

- Логирование запросов
- Обработка ошибок
- Метрики производительности

## 🚀 Развертывание

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
```


---

**Удачной разработки! 🚀** 