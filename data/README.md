# Данные

Данные не хранятся в репозитории: они большие и распространяются по лицензиям источников. Скачайте их сами и разложите так:

```
data/
├── pad_ufes_20/
│   ├── metadata.csv
│   └── images/imgs_part_1/*.png, imgs_part_2/*.png, imgs_part_3/*.png
└── isic2024/
    ├── train-metadata.csv
    └── train-image.hdf5
```

Точная вложенность папок с картинками не важна: снимки ищутся рекурсивно по имени файла из столбца `img_id`.

## 1. PAD-UFES-20 (фото со смартфона + анкета)

- Источник: Mendeley Data, https://data.mendeley.com/datasets/zr7vgbcyr2/1
- Статья: Pacheco A. G. C. et al. *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones.* Data in Brief, 32, 106221 (2020).
- Лицензия: CC BY 4.0.

Шаги:

1. Скачайте архив со страницы набора (кнопка «Download All»).
2. Распакуйте его в `data/pad_ufes_20/`.
3. **Распакуйте вложенные архивы** `images/imgs_part_1.zip`, `imgs_part_2.zip`, `imgs_part_3.zip`, если они есть. Это частая причина ошибки «не найдено файлов».

## 2. ISIC 2024 / SLICE-3D (400 000+ снимков смартфонного качества)

- Источник: Kaggle, соревнование ISIC 2024, https://www.kaggle.com/competitions/isic-2024-challenge
- Статья: Kurtansky N. R. et al. *The SLICE-3D dataset.* Scientific Data, 11, 884 (2024).
- Лицензия: своя для каждого снимка (столбец `copyright_license` в метаданных: варианты CC BY, CC BY-NC, CC0). Для учебного проекта используйте данные некоммерчески и указывайте авторство.

Шаги:

1. Войдите на Kaggle и примите правила соревнования на вкладке **Rules**, иначе скачать не получится.
2. Создайте API-токен: Kaggle → Settings → API → Create New Token. Положите `kaggle.json` в `~/.kaggle/` (в Windows: `C:\Users\<имя>\.kaggle\`).
3. Скачайте только два нужных файла:

```bash
pip install kaggle
kaggle competitions download -c isic-2024-challenge -f train-metadata.csv -p data/isic2024
kaggle competitions download -c isic-2024-challenge -f train-image.hdf5 -p data/isic2024
```

4. Если файлы скачались как `.zip`, распакуйте их в `data/isic2024/`.

## Проверка

```bash
python scripts/check_data.py
```

Скрипт покажет число снимков, пациентов, диагнозов и разбиение по пациентам, а при проблеме объяснит, что не так.
