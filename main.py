from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json

app = Flask(__name__)


@app.route("/")
def root():
    """
    Функция обработчика главной страницы сайта.
    :return: Содержимое файла index.html
    """
    with open("./index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Обработчик конечной точки /detect POST
    Получает загруженный файл с именем "image_file",
    пропускает его через сеть YOLOv8
    и возвращает массив ограничивающих рамок.
    :return: массив JSON с объектами, ограничивающих рамок в формате
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(json.dumps(boxes), mimetype="application/json")


def detect_objects_on_image(buf):
    """
    Функция получает изображение,
    пропускает его через нейронную сеть YOLOv8
    и возвращает массив обнаруженных объектов
    и ограничивающих их рамок
    :param buf: Входной поток файла изображения
    :return: Массив ограничивающих рамок в формате
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO(r".\runs\yolov8n_candle_detection\weights\best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
