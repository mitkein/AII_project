#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс для детекции дефектов
Работает в браузере, не требует дополнительных GUI библиотек
"""

import os
import sys
import numpy as np
from PIL import Image
import base64
import io
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json

# Импорты для модели
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    print("❌ TensorFlow не установлен")
    sys.exit(1)

class DetectorModel:
    def __init__(self):
        self.binary_model = None
        self.class_names = ['🚨 Дефекты обнаружены', '✅ Дефекты не обнаружены']
        self.load_model()
        
    def load_model(self):
        """Загрузка модели"""
        try:
            binary_model_path = "C:\Users\ооп\best_model_binary.h5"
            if os.path.exists(binary_model_path):
                print("⏳ Загружаю бинарную модель...")
                self.binary_model = load_model(binary_model_path, compile=False)
                print("✅ Модель загружена!")
                return True
            else:
                print("❌ Файл best_model_binary.h5 не найден!")
                return False
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
            
    def preprocess_image(self, image_data):
        """Предобработка изображения"""
        try:
            print(f"🔄 Начинаю предобработку изображения, размер: {len(image_data)} байт")
            
            # Конвертируем base64 в изображение
            img = Image.open(io.BytesIO(image_data))
            print(f"📐 Размер исходного изображения: {img.size}, режим: {img.mode}")
            
            # Конвертируем в RGB если нужно
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"🔄 Конвертировал в RGB")
            
            # Изменяем размер и конвертируем в grayscale для модели
            img_resized = img.resize((400, 400), Image.Resampling.LANCZOS)
            img_gray = img_resized.convert('L')  # Конвертируем в grayscale
            print(f"📐 Изменен размер до: {img_gray.size}")
            
            # Конвертируем в numpy array
            img_array = np.array(img_gray, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)  # Добавляем batch dimension
            img_array = np.expand_dims(img_array, axis=-1)  # Добавляем channel dimension
            img_array /= 255.0  # Нормализация
            
            print(f"✅ Предобработка завершена, форма массива: {img_array.shape}")
            return img_array
            
        except Exception as e:
            print(f"❌ Ошибка предобработки: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def analyze_image(self, image_data):
        """Анализ изображения"""
        print(f"🔍 Начинаю анализ изображения...")
        
        if not self.binary_model:
            print("❌ Модель не загружена")
            return {"error": "Модель не загружена"}
            
        try:
            # Предобработка
            print("🔄 Предобработка изображения...")
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                print("❌ Ошибка предобработки")
                return {"error": "Ошибка предобработки изображения"}
                
            # Анализ
            print("🧠 Запускаю модель...")
            prediction = self.binary_model.predict(img_array, verbose=0)
            print(f"📊 Получено предсказание: {prediction}")
            print(f"📊 Форма предсказания: {prediction.shape}")
            
            # Обработка результата
            if len(prediction.shape) > 1 and prediction.shape[1] == 2:  # Categorical
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                print(f"📈 Categorical: класс {class_idx}, уверенность {confidence}")
            else:  # Binary
                confidence = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
                class_idx = 1 if confidence > 0.5 else 0
                print(f"📈 Binary: класс {class_idx}, уверенность {confidence}")
                
            result = {
                "class": self.class_names[class_idx],
                "class_idx": int(class_idx),
                "confidence": float(confidence),
                "confidence_percent": float(confidence * 100),
                "interpretation": self.generate_interpretation(class_idx, confidence)
            }
            
            print(f"✅ Анализ завершен: {result['class']}")
            return result
            
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Ошибка анализа: {str(e)}"}
            
    def generate_interpretation(self, class_idx, confidence):
        """Генерация интерпретации"""
        if class_idx == 0:  # Дефекты обнаружены
            base_text = "Модель обнаружила признаки дефектов на изображении."
            if confidence > 0.9:
                return f"{base_text} ОЧЕНЬ ВЫСОКАЯ уверенность (>90%). Рекомендация: изделие требует отбраковки."
            elif confidence > 0.8:
                return f"{base_text} ВЫСОКАЯ уверенность (80-90%). Рекомендация: изделие следует отбраковать."
            elif confidence > 0.6:
                return f"{base_text} СРЕДНЯЯ уверенность (60-80%). Рекомендация: требуется дополнительная проверка."
            else:
                return f"{base_text} НИЗКАЯ уверенность (50-60%). Рекомендация: необходима ручная проверка."
        else:  # Дефекты не обнаружены
            base_text = "Модель не обнаружила признаков дефектов на изображении."
            if confidence > 0.9:
                return f"{base_text} ОЧЕНЬ ВЫСОКАЯ уверенность (>90%). Рекомендация: изделие можно пропустить."
            elif confidence > 0.8:
                return f"{base_text} ВЫСОКАЯ уверенность (80-90%). Рекомендация: изделие в хорошем состоянии."
            elif confidence > 0.6:
                return f"{base_text} СРЕДНЯЯ уверенность (60-80%). Рекомендация: возможна дополнительная проверка."
            else:
                return f"{base_text} НИЗКАЯ уверенность (50-60%). Рекомендация: рекомендуется контроль."

# Глобальный экземпляр модели
detector = DetectorModel()

class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            html_content = self.get_html_page()
            self.wfile.write(html_content.encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_OPTIONS(self):
        """Обработка OPTIONS запросов для CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def log_message(self, format, *args):
        """Подавляем стандартные логи сервера"""
        pass
            
    def do_POST(self):
        """Обработка POST запросов"""
        if self.path == '/analyze':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                print(f"📥 Получен POST запрос, размер: {content_length} байт")
                
                # Парсим данные
                data = json.loads(post_data.decode('utf-8'))
                image_data_base64 = data['image'].split(',')[1]  # Убираем "data:image/jpeg;base64,"
                image_data = base64.b64decode(image_data_base64)
                
                print(f"🖼️ Размер изображения: {len(image_data)} байт")
                
                # Анализируем
                result = detector.analyze_image(image_data)
                print(f"📊 Результат анализа: {result}")
                
                # Отправляем результат
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response_json = json.dumps(result, ensure_ascii=False)
                self.wfile.write(response_json.encode('utf-8'))
                
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка JSON: {e}")
                self.send_error_response(f"Ошибка JSON: {str(e)}")
            except Exception as e:
                print(f"❌ Ошибка анализа: {e}")
                self.send_error_response(f"Ошибка анализа: {str(e)}")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = json.dumps({"error": "Неизвестный путь"})
            self.wfile.write(error_response.encode('utf-8'))
            
    def send_error_response(self, error_message):
        """Отправка ошибки в формате JSON"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = json.dumps({"error": error_message}, ensure_ascii=False)
        self.wfile.write(error_response.encode('utf-8'))
            
    def get_html_page(self):
        """HTML страница приложения"""
        return """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Детектор дефектов</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #007bff;
        }
        .upload-area.dragover {
            border-color: #007bff;
            background-color: #f0f8ff;
        }
        #imagePreview {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #0056b3;
        }
        .button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result-success {
            color: #28a745;
            font-weight: bold;
        }
        .result-danger {
            color: #dc3545;
            font-weight: bold;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background-color: #007bff;
            transition: width 0.3s;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Детектор дефектов</h1>
        <p>Анализ изображений на наличие дефектов</p>
    </div>

    <div class="container">
        <!-- Левая панель - управление -->
        <div class="panel">
            <h3>Управление</h3>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📂 Нажмите или перетащите изображение сюда</p>
                <p style="font-size: 12px; color: #666;">
                    Поддерживаемые форматы: JPG, PNG, BMP, TIFF
                </p>
            </div>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="button" id="analyzeBtn" onclick="analyzeImage()" disabled>
                    🔍 Анализировать
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Анализирую изображение...</p>
            </div>
        </div>

        <!-- Центральная панель - изображение -->
        <div class="panel">
            <h3>Изображение</h3>
            <div style="text-align: center;">
                <img id="imagePreview" style="display: none;">
                <p id="noImageText" style="color: #666; margin-top: 100px;">
                    Изображение не загружено
                </p>
                <p id="fileName" style="font-size: 12px; color: #666; margin-top: 10px;"></p>
            </div>
        </div>

        <!-- Правая панель - результаты -->
        <div class="panel">
            <h3>Результаты анализа</h3>
            
            <div style="margin: 20px 0;">
                <strong>Результат:</strong>
                <div id="result" style="font-size: 18px; margin: 10px 0; color: #666;">
                    Не проанализировано
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                <strong>Уверенность:</strong>
                <div id="confidence" style="margin: 10px 0;">-</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                <strong>Интерпретация:</strong>
                <div id="interpretation" style="margin-top: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 14px; line-height: 1.4;">
                    Загрузите изображение и нажмите "Анализировать" для получения результата.
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImage = null;

        // Обработка загрузки файла
        document.getElementById('fileInput').addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });

        // Drag & Drop
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('Пожалуйста, выберите изображение');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = e.target.result;
                
                // Показываем изображение
                const preview = document.getElementById('imagePreview');
                const noImageText = document.getElementById('noImageText');
                const fileName = document.getElementById('fileName');
                
                preview.src = currentImage;
                preview.style.display = 'block';
                noImageText.style.display = 'none';
                fileName.textContent = file.name;
                
                // Активируем кнопку анализа
                document.getElementById('analyzeBtn').disabled = false;
                
                // Очищаем предыдущие результаты
                clearResults();
            };
            reader.readAsDataURL(file);
        }

        function clearResults() {
            document.getElementById('result').textContent = 'Не проанализировано';
            document.getElementById('result').className = '';
            document.getElementById('confidence').textContent = '-';
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('interpretation').textContent = 'Нажмите "Анализировать" для получения результата.';
        }

        async function analyzeImage() {
            if (!currentImage) {
                alert('Сначала загрузите изображение');
                return;
            }

            // Показываем загрузку
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            
            console.log('🔍 Начинаю анализ изображения...');

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: currentImage
                    })
                });

                console.log('📡 Ответ сервера получен, статус:', response.status);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    const text = await response.text();
                    console.error('❌ Сервер вернул не JSON:', text);
                    throw new Error('Сервер вернул некорректный ответ (не JSON)');
                }

                const result = await response.json();
                console.log('📊 Результат анализа:', result);

                if (result.error) {
                    alert('Ошибка анализа: ' + result.error);
                    return;
                }

                // Отображаем результаты
                const resultElement = document.getElementById('result');
                resultElement.textContent = result.class;
                resultElement.className = result.class_idx === 0 ? 'result-danger' : 'result-success';
                
                document.getElementById('confidence').textContent = 
                    `${result.confidence.toFixed(4)} (${result.confidence_percent.toFixed(1)}%)`;
                
                document.getElementById('progressFill').style.width = result.confidence_percent + '%';
                
                document.getElementById('interpretation').textContent = result.interpretation;

            } catch (error) {
                console.error('❌ Ошибка:', error);
                alert('Ошибка соединения: ' + error.message);
            } finally {
                // Скрываем загрузку
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
        """

def run_server(port=8080):
    """Запуск веб-сервера"""
    try:
        server = HTTPServer(('localhost', port), WebHandler)
        print(f"🌐 Веб-сервер запущен на http://localhost:{port}")
        print("🚀 Открываю браузер...")
        
        # Открываем браузер
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
        
        print("📱 Для остановки нажмите Ctrl+C")
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n👋 Остановка сервера...")
        server.shutdown()
    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")

def main():
    """Главная функция"""
    print("=" * 60)
    print("🌐 ВЕБЛЕКТОР ДЕФЕКТОВ - ВЕБ-ИНТЕРФЕЙС")
    print("=" * 60)
    
    if not detector.binary_model:
        print("❌ Модель не загружена. Проверьте файл best_model_binary.h5")
        sys.exit(1)
        
    print("✅ Модель готова к работе!")
    
    # Запускаем сервер
    try:
        run_server()
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()
