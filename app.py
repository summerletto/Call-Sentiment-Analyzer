import streamlit as st
from openai import OpenAI
import docx
import requests
import io
import time
from datetime import datetime
from config import OPENROUTER_API_KEY, HUGGINGFACE_TOKEN

# -------------------- Конфигурация --------------------
st.set_page_config(
    page_title="Анализатор тона звонков",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Агент оценки тона телефонных разговоров")
st.markdown("""
Загрузите текстовую расшифровку разговора (текст или файл), и ИИ определит его общий тон и даст рекомендации по улучшению.
*Работает на платформе OpenRouter и модели DeepSeek V3.*
""")

# Инициализация клиента OpenRouter
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
def create_test_docx_file():
    """Создает тестовый DOCX файл с примером разговора."""
    doc = docx.Document()

    # Добавляем заголовок
    doc.add_heading('Пример телефонного разговора', 0)

    # Добавляем абзацы с текстом
    doc.add_paragraph('Дата: ' + datetime.now().strftime("%d.%m.%Y"))
    doc.add_paragraph('')
    doc.add_paragraph('Оператор: Здравствуйте! Компания "Сервис Плюс". Меня зовут Анна. Чем могу помочь?')
    doc.add_paragraph('')
    doc.add_paragraph('Клиент: Добрый день. У меня проблема с интернетом - он очень медленно работает уже второй день.')
    doc.add_paragraph('')
    doc.add_paragraph(
        'Оператор: Понимаю ваше недовольство. Давайте проверим соединение. Вы пробовали перезагрузить роутер?')
    doc.add_paragraph('')
    doc.add_paragraph('Клиент: Да, пробовал, не помогло. Это очень раздражает, я не могу работать!')
    doc.add_paragraph('')
    doc.add_paragraph(
        'Оператор: Конечно, понимаю ваше разочарование. Я создам заявку для наших технических специалистов. Они свяжутся с вами в течение часа.')
    doc.add_paragraph('')
    doc.add_paragraph('Клиент: Спасибо. Надеюсь, это решит проблему.')
    doc.add_paragraph('')
    doc.add_paragraph('Оператор: Обязательно решим! Хорошего дня!')

    # Сохраняем документ в байтовый поток
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)

    return doc_io


def query(payload):
    """Отправляет запрос к Hugging Face API"""
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def extract_text_from_file(uploaded_file):
    """Извлекает текст из загруженного файла."""
    try:
        # Сохраняем позицию файла
        current_position = uploaded_file.tell()
        uploaded_file.seek(0)

        # Для текстовых файлов
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            # Пробуем разные кодировки
            encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode(encoding)
                    return content
                except UnicodeDecodeError:
                    continue

            st.error("Не удалось декодировать файл. Попробуйте сохранить файл в UTF-8 кодировке.")
            return None

        # Для DOCX файлов
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])

        else:
            st.error("Неподдерживаемый формат файла. Поддерживаются только .txt и .docx файлы.")
            return None

    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None
    finally:
        # Всегда возвращаем позицию файла в исходное состояние
        uploaded_file.seek(current_position)


def analyze_sentiment_with_api(text):
    """Анализирует тональность текста с помощью Hugging Face API."""
    if not HUGGINGFACE_TOKEN:
        return {"error": "Токен Hugging Face не настроен. Проверьте файл конфигурации."}

    try:
        # Отправляем запрос к API
        output = query({
            "inputs": text[:512]  # Обрезаем текст до первых 512 символов
        })

        # Обрабатываем ответ API
        if isinstance(output, list) and len(output) > 0:
            # Находим запись с наибольшей уверенностью
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

    # Список моделей в порядке приоритета
    models = [
        "deepseek/deepseek-chat-v3-0324:free",
        "qwen/qwen3-30b-a3b:free",
        "google/gemini-2.0-flash-thinking-exp:free",
    ]

    last_error = None

    for index, model_name in enumerate(models):
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
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )

            # Проверяем, что ответ не пустой
            response = completion.choices[0].message.content.strip()
            if response:
                return response
            else:
                last_error = "Модель вернула пустой ответ"
                continue

        except Exception as e:
            error_msg = str(e)
            last_error = error_msg

            # Проверяем различные варианты ошибки лимита запросов
            rate_limit_indicators = [
                "429",
                "rate limit",
                "too many requests",
                "limit exceeded",
                "overload",
                "busy",
                " temporarily "
            ]

            is_rate_limit_error = any(indicator in error_msg for indicator in rate_limit_indicators)

            # Если это ошибка лимита и есть еще модели для попытки
            if is_rate_limit_error and index < len(models) - 1:
                # Создаем элемент для отображения предупреждения
                warning_placeholder = st.empty()
                warning_placeholder.warning(
                    f"Модель {model_name} временно недоступна (лимит запросов). Пробую следующую модель через 2 секунды...")
                time.sleep(2)
                warning_placeholder.empty()  # Убираем предупреждение после задержки
                continue

            # Если это другая ошибка или это последняя модель в списке
            if index == len(models) - 1:
                break  # Прерываем цикл, если это последняя модель

    # Если дошли до этой точки, значит все модели не сработали
    return f"Не удалось получить рекомендации. Последняя ошибка: {last_error}"


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
    - Генерация рекомендаций: DeepSeek V3 через OpenRouter 
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
        # Показываем информацию о файле для отладки
        st.write(f"Имя файла: {uploaded_file.name}")
        st.write(f"Тип файла: {uploaded_file.type}")
        st.write(f"Размер файла: {uploaded_file.size} байт")

        input_text = extract_text_from_file(uploaded_file)
        if input_text:
            # Создаем уникальный ключ на основе имени файла и его размера
            unique_key = f"file_content_{uploaded_file.name}_{uploaded_file.size}"
            st.text_area("Текст из файла:", value=input_text, height=250, key=unique_key)
        else:
            st.error("Не удалось извлечь текст из файла. Убедитесь, что файл не пустой и имеет правильную кодировку.")

if st.button("Проанализировать", type="primary"):
    if not input_text or not input_text.strip():
        st.warning("Пожалуйста, введите текст для анализа или загрузите файл.")
        st.stop()

    # Шаг 1: Анализ тональности через API
    with st.spinner("Анализируем общий тон разговора через Hugging Face API..."):
        sentiment_result = analyze_sentiment_with_api(input_text)

    if "error" in sentiment_result:
        st.error(sentiment_result["error"])
        st.stop()

    # Отображаем результат анализа тональности
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    # Переводим метки на русский для лучшего восприятия
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

    # Закрываем блок колонок перед генерацией рекомендаций
    # Шаг 2: Генерация рекомендаций через LLM
    with st.spinner("Генерируем персонализированные рекомендации с помощью ИИ..."):
        recommendations = generate_llm_recommendations(input_text, sentiment_label_ru)

    # Проверяем, есть ли рекомендации
    if recommendations and not recommendations.startswith("Не удалось получить рекомендации"):
        st.subheader("🎯 Рекомендации по улучшению:")
        st.markdown(recommendations)
    else:
        st.error(recommendations)

# Кнопки для создания тестовых файлов
col1, col2 = st.columns(2)

with col1:
    # Кнопка для создания тестового TXT файла
    test_text = "Это тестовый разговор. Все прошло хорошо, клиент был доволен."
    st.download_button(
        label="Скачать тестовый TXT файл",
        data=test_text,
        file_name="тестовый_разговор.txt",
        mime="text/plain"
    )

with col2:
    # Кнопка для создания тестового DOCX файла
    docx_file = create_test_docx_file()
    st.download_button(
        label="Скачать тестовый DOCX файл",
        data=docx_file,
        file_name="тестовый_разговор.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

# -------------------- Информационный блок --------------------

with st.expander("ℹ️ О данном приложении"):
    st.markdown("""
    Это тестовое приложение **ИИ-агента для оценки телефонных звонков**, созданное в рамках тестового задания.

    **Особенности:**
    - **Анализ тональности:** Выполняется через Hugging Face Inference API.
    - **Генерация рекомендаций:** Выполняется с помощью модели DeepSeek V3 через OpenRouter.
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
        2. Убедитесь, что на вашем аккаунте OpenRouter есть кредиты 
        3. При ошибке 429 (ограничение скорости) приложение автоматически переключится на другую модель

        **Используемые модели (в порядке приоритета):**
        - deepseek/deepseek-chat-v3-0324:free (основная)
        - qwen/qwen3-30b-a3b:free (резервная)
        - google/gemini-2.0-flash-thinking-exp:free (резервная)

        **Коды ошибок OpenRouter:** 
        - 400: Неверный запрос
        - 401: Неверные учетные данные
        - 402: Недостаточно кредитов
        - 403: Модерация отклонена
        - 408: Таймаут запроса
        - 429: Превышен лимит запросов (ограничение скорости)
        - 502: Ошибка модели
        - 503: Нет доступных провайдеров
        """)