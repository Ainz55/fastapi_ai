from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
import os

# Создаем экземпляр FastAPI
app = FastAPI(title="AI Project Website",
              description="Сайт для интеграции ИИ проекта")

# Настраиваем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настраиваем шаблоны
templates = Jinja2Templates(directory="templates")

# Создаем директории если их нет
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ai-demo", response_class=HTMLResponse)
async def ai_demo(request: Request):
    """Страница демонстрации ИИ"""
    return templates.TemplateResponse("ai_demo.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """Страница о проекте"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/api/process-text")
async def process_text(text: str = Form(...)):
    """API endpoint для обработки текста ИИ"""
    # Здесь будет ваша логика ИИ
    return {
        "input": text,
        "output": f"Обработанный текст: {text}",
        "status": "success"
    }


@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """API endpoint для загрузки файлов"""
    # Здесь будет ваша логика обработки файлов ИИ
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {
        "filename": file.filename,
        "size": len(content),
        "status": "uploaded"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
