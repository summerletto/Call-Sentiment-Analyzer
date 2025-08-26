import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Получаем ключи из переменных окружения
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
