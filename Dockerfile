# Веб-интерфейс нормирования ШАУ
FROM python:3.11-slim

# Системные зависимости pdfplumber (pdfminer.six тянет шрифты/parsing — slim хватает)
WORKDIR /app

# Сначала зависимости — для кэширования слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код
COPY . .

EXPOSE 8000

# Контейнер слушает на всех интерфейсах (наружу пробрасывается через -p)
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
