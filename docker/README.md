# Segment Anything (SAM) — Docker-демо

Docker-демо для модели Segment Anything от Meta AI. Выделите объект, указав точку внутри него. Результат сохраняется как чёрно-белая маска и как исходное изображение с цветной маской поверх.

- Статья: https://arxiv.org/abs/2304.02643
- Официальный код и инструкции: https://github.com/facebookresearch/segment-anything
- Выбранная модель: `vit_b`; официальный чекпойнт `sam_vit_b_01ec64.pth` скачивается при первом запуске.

В этом проекте официальный SAM устанавливается из GitHub на закреплённом коммите `dca509fe793f601edb92606367a655c15ac00fdf`. Контейнер использует CPU-версию PyTorch и запускается без NVIDIA GPU.

## Требования

- Windows 10/11, Git и Docker Desktop в режиме Linux containers.
- Интернет при сборке и первом запуске.
- Несколько гигабайт свободного места и оперативной памяти.

## Установка на новом ПК

После создания форка в GitHub Docker-проект будет находиться в каталоге `docker`. В PowerShell выполните:

```powershell
git clone https://github.com/collserra21/segment-anything.git
cd segment-anything\docker
docker build -t sam-point-demo .
New-Item -ItemType Directory -Force output | Out-Null
docker run --rm -v "${PWD}/output:/app/output" -v sam-model-cache:/models sam-point-demo --make-demo-image
```

Первый запуск скачивает чекпойнт объёмом около 375 МБ. Последующие запуски используют Docker volume `sam-model-cache` и не скачивают его повторно.

## Результаты

В папке `output` появятся:

- `overlay.png` — демо-изображение с красной сегментацией и зелёной точкой запроса;
- `mask.png` — чёрно-белая маска объекта;
- `result.json` — координаты точки, площадь маски и оценка модели.

## Сегментация собственного изображения

Положите изображение в папку `docker` под именем `my-photo.jpg`. Укажите координату точки внутри нужного объекта; начало координат — левый верхний угол изображения:

```powershell
docker run --rm -v "${PWD}/output:/app/output" -v sam-model-cache:/models -v "${PWD}/my-photo.jpg:/input/photo.jpg:ro" sam-point-demo --image /input/photo.jpg --point-x 420 --point-y 260
```

Замените `420` и `260` координатами точки внутри объекта. Файлы результата в `output` будут перезаписаны.

## Тесты

```powershell
docker run --rm -v "${PWD}/output:/app/output" -v sam-model-cache:/models --entrypoint pytest sam-point-demo -q
```

Тесты запускают модель, проверяют, что она создала непустую маску, и сохраняют `overlay.png`, `mask.png` и `result.json` в `output`. На первом запуске тест также скачает чекпойнт.

## Лицензия и область проверки

Официальный код SAM и модель SAM лицензированы по Apache 2.0. Демо запускает инференс на тестовой картинке, но не повторяет полное обучение или оценивание статьи на датасете SA-1B.
