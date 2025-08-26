import streamlit as st
from openai import OpenAI  # Импортируем из нового местоположения
import docx
import requests
from config import OPENROUTER_API_KEY, HUGGINGFACE_TOKEN

# -------------------- Конфигурация --------------------
# Настройка страницы Streamlit
st.set_page_config(
    page_title="Анализатор тона звонков",
    page_icon="📞",
    layout="wide"
)

# Заголовок приложения
st.title("📞 Агент оценки тона телефонных разговоров")
st.markdown("""
Загрузите текстовую расшифровку разговора (текст или файл), и ИИ определит его общий тон и даст рекомендации по улучшению.
*Работает на платформе OpenRouter и модели DeepSeek V3.*
""")

# Инициализация клиента OpenRouter :cite[2]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# URL для Hugging Face API
API_URL = "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
}


# -------------------- Функции --------------------
def query(payload):
    """Отправляет запрос к Hugging Face API"""
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def extract_text_from_file(uploaded_file):
    """Извлекает текст из загруженного файла."""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error("Неподдерживаемый формат файла. Поддерживаются только .txt и .docx файлы.")
            return None
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None


def analyze_sentiment_with_api(text):
    """Анализирует тональность текста с помощью Hugging Face API."""
    if not HUGGINGFACE_TOKEN:
        return {"error": "Токен Hugging Face не настроен. Проверьте файл конфигурации."}

    try:
        output = query({"inputs": text[:512]})

        if isinstance(output, list) and len(output) > 0:
            most_confident = max(output[0], key=lambda x: x['score'])
            return most_confident
        elif isinstance(output, dict) and 'error' in output:
            return {"error": output['error']}
        else:
            return {"error": "Неожиданный формат ответа от API"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка при обращении к Hugging Face API: {e}"}
    except Exception as e:
        return {"error": f"Ошибка анализа тональности: {e}"}


def generate_llm_recommendations(text, sentiment_label):
    """Генерирует рекомендации через OpenRouter API на основе текста и тона."""
    if not OPENROUTER_API_KEY:
        return "Ошибка: API ключ OpenRouter не настроен. Проверьте файл конфигурации."

    try:
        # Промпт для генерации рекомендаций
        prompt = f"""
        Ты — опытный бизнес-тренер по коммуникациям. Проанализируй расшифровку телефонного разговора и дай рекомендации.

        ТОН РАЗГОВОРА: {sentiment_label}
        ТЕКСТ РАЗГОВОРА: 
        \"\"\"{text}\"\"\"

        Дай 1-2 конкретные, практические рекомендации на русском языке, как улучшить подобные разговоры в будущем.
        Ответ выдай в виде маркированного списка, без лишних вступлений.
        """

        # Вызов OpenRouter API
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Ошибка при генерации рекомендаций: {e}"

# -------------------- Интерфейс --------------------
with st.sidebar:
    st.header("Конфигурация")
    if OPENROUTER_API_KEY:
        st.success("API ключ OpenRouter загружен")
    else:
        st.error("API ключ OpenRouter не найден")

    if HUGGINGFACE_TOKEN:
        st.success("Токен Hugging Face загружен")
    else:
        st.warning("Токен Hugging Face не найден")

    st.markdown("""
    **Поддерживаемые форматы файлов:**
    - Текстовые файлы (.txt)
    - Документы Word (.docx)

    **Используемые сервисы:**
    - Анализ тональности: Hugging Face Inference API
    - Генерация рекомендаций: DeepSeek V3 через OpenRouter :cite[1]
    """)

# Основной интерфейс
input_method = st.radio("Выберите способ ввода:", ["Текст", "Файл"])

input_text = ""

if input_method == "Текст":
    input_text = st.text_area("Введите расшифровку телефонного разговора:", height=250,
                              placeholder="Скопируйте текст расшифровки сюда...")
else:
    uploaded_file = st.file_uploader("Загрузите файл с расшифровкой", type=["txt", "docx"])
    if uploaded_file is not None:
        input_text = extract_text_from_file(uploaded_file)
        if input_text:
            st.text_area("Текст из файла:", value=input_text, height=250)

if st.button("Проанализировать", type="primary"):
    if not input_text.strip():
        st.warning("Пожалуйста, введите текст для анализа.")
        st.stop()

    # Шаг 1: Анализ тональности
    with st.spinner("Анализируем общий тон разговора через Hugging Face API..."):
        sentiment_result = analyze_sentiment_with_api(input_text)

    if "error" in sentiment_result:
        st.error(sentiment_result["error"])
        st.stop()

    # Отображаем результат анализа тональности
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    sentiment_labels_ru = {
        "POSITIVE": "ПОЗИТИВНЫЙ",
        "NEGATIVE": "НЕГАТИВНЫЙ",
        "NEUTRAL": "НЕЙТРАЛЬНЫЙ",
        "LABEL_0": "НЕГАТИВНЫЙ",
        "LABEL_1": "НЕЙТРАЛЬНЫЙ",
        "LABEL_2": "ПОЗИТИВНЫЙ"
    }

    sentiment_label_ru = sentiment_labels_ru.get(sentiment_label, sentiment_label)

    st.success("Анализ тональности завершен!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="**Преобладающий тон**", value=sentiment_label_ru)
    with col2:
        st.metric(label="**Уверенность модели**", value=f"{sentiment_score:.2f}")

    # Шаг 2: Генерация рекомендаций через LLM
    with st.spinner("Генерируем персонализированные рекомендации с помощью ИИ..."):
        recommendations = generate_llm_recommendations(input_text, sentiment_label_ru)

    st.subheader("🎯 Рекомендации по улучшению:")
    st.markdown(recommendations)

# -------------------- Информационный блок --------------------
with st.expander("ℹ️ О данном приложении"):
    st.markdown("""
    Это тестовое приложение **ИИ-агента для оценки телефонных звонков**, созданное в рамках тестового задания.

    **Особенности:**
    - **Анализ тональности:** Выполняется через Hugging Face Inference API.
    - **Генерация рекомендаций:** Выполняется с помощью модели DeepSeek V3 через OpenRouter :cite[1].
    - **Поддержка файлов:** Работа с текстовыми файлами (.txt) и документами Word (.docx).
    - **Безопасность:** Ключи API хранятся в отдельном файле, исключенном из системы контроля версий.

    **Технологии:** Python, Streamlit, Hugging Face Inference API, OpenRouter API, python-docx
    """)

with st.expander("⚠️ Устранение неполадок"):
    st.markdown("""
    **Если анализ тональности не работает:**
    1. Убедитесь, что в файле `.env` указан правильный токен Hugging Face
    2. Проверьте, что на вашем аккаунте Hugging Face есть доступ к Inference API

    **Если генерация рекомендаций не работает:**
    1. Проверьте правильность OpenRouter API ключа в файле `.env`
    2. Убедитесь, что на вашем аккаунте OpenRouter есть кредиты :cite[10]
    3. Проверьте, что модель `deepseek/deepseek-chat-v3-0324:free` доступна :cite[1]

    **Коды ошибок OpenRouter:** :cite[3]
    - 400: Неверный запрос
    - 401: Неверные учетные данные
    - 402: Недостаточно кредитов
    - 403: Модерация отклонена
    - 408: Таймаут запроса
    - 429: Превышен лимит запросов
    - 502: Ошибка модели
    - 503: Нет доступных провайдеров
    """)