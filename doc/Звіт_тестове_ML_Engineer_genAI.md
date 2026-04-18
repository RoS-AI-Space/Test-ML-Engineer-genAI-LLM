# Звіт: автоматична сегментація мобільних додатків на субніші (ML Engineer, genAI)

Цей файл — **єдиний структурований звіт** за п. 6 ТЗ до документа `Тестове завдання на ML Engineer genAI (LLM).md`: код і логіка, обґрунтування підходу, аналіз результатів, формат виводу та посилання на матеріали (п. 7).

---

## Контекст з ТЗ

- **Мета:** вузькі **субніші** з **прямими конкурентами** всередині кожної групи.
- **Вхід:** ~4200 записів; поля зокрема `trackName`, `description`, `screenshotUrls`, `overview`, `features`.
- **Вихід:** JSON (назва ніші, опис, список конкурентів) + табличний маппінг; якість підписів критична.
- **Оцінка:** гранулярність і змістовність ніш; допускається **лише текст**, скріншоти — опційно.

Додаткові дані / рерайт з ТЗ (рядок про розширення) для мінімального рішення **не обов’язкові** — використовуються `overview` та `features`.

---

## 1. Код і логіка

### 1.1 Файли проєкту

| Компонент | Шлях |
| --- | --- |
| Пайплайн | `code/segment.py` |
| Конфігурація | `code/config.py` |
| Допоміжний batch-enrichment (локально; може бути в `.gitignore`) | `code/enrich_batch.py` |
| Пост-рефайнмент після основного прогону | `code/refine_niches.py` |
| Приклад `.env` | `code/.env.example` |

Enrichment у продакшн-гілці репозиторію виконується **`segment.py` (Stage 1)**. `enrich_batch.py` — окремий допоміжний сценарій; у клоні репо файлу може не бути через ignore.

У `segment.py` задокументовано **Stage 0–6** (docstring на початку файлу).

### 1.2 Етапи `segment.py`

1. **Stage 0 — дані**  
   Завантаження JSON → `DataFrame`; `clean_text`; порожній `overview` → початок `description`; `features` у один рядок; **дедуплікація назв** (rapidfuzz, поріг 92).

2. **Stage 1 — LLM enrichment (опційно)**  
   Поля `primary_jtbd`, `target_user`, `core_value`, `category_narrow` у вигляді **markdown-таблиці**; парсер у `segment.py`; паралель і таймаути з `config`; збереження в `code/result/cache/enrichments.json`. Пропуск: немає ключа / прапорці `--no-llm` або `--no-llm-enrichment`.

3. **Канонічний текст**  
   Збирається з `trackName`, `overview`, `features` і (якщо є) enrichment — орієнтир на **призначення продукту** для конкурентної близькості в ембеддингах.

4. **Stage 2 — ембеддинги**  
   `config.get_embedding_chain()`: за замовчуванням **тільки локально** `BAAI/bge-small-en-v1.5` (SentenceTransformers). Якщо в `.env` задано `EMBEDDING_MODEL` / `EMBEDDING_FALLBACK_MODEL` — спочатку **API (litellm, зокрема OpenRouter)**, потім другий API за наявності, **останнім** — локальний bge. Вектори **нормалізовані**.

5. **Stage 3 — кластеризація**  
   **AgglomerativeClustering** (complete linkage, cosine), адаптивні пороги за вимірністю; рекурсивне дроблення великих кластерів, перепризначення шуму, злиття схожих центроїдів (**coarse-to-fine**).

6. **Stage 4 — неймінг**  
   LLM батчами або **rule-based** fallback, якщо LLM вимкнено / недоступний.

7. **Stage 5–6**  
   Метрики (`compute_metrics`: silhouette по inlier-ах, cosine; середня внутрішньокластерна схожість на підвибірці кластерів); експорт `niches.json`, `app_niche_mapping.csv`, `metrics.json`; **UMAP + Plotly** → `umap_clusters.html`.

### 1.3 Структура коду

Модульні етапи дозволяють відтворювати прогін і робити абляції (`--no-llm`, `--no-llm-enrichment`, `--no-llm-naming`). Конфігурація винесена в `config.py` і `.env` без секретів у коді.

---

## 2. Підхід і альтернативи

**Конкурентна схожість:** ембеддинги + опційне LLM-збагачення + ієрархічна кластеризація без фіксованого K.

**Чому не K-Means:** потрібен K, нерівномірні ніші. **Чому не DBSCAN як основа:** чутливість до масштабу відстаней на текстових ембеддингах; тут — контрольований пороговий agglomerative + постобробка.

**LLM:** OpenRouter / ланцюг fallback з `.env`; неймінг без LLM дає шаблонніші назви (див. окремий прогін `--no-llm`).

**Без пікселів `screenshotUrls`:** відповідає ТЗ «лише текст»; ASO-частина вже в текстових полях.

---

## 3. Результати та артефакти

### 3.1 Формат (як у ТЗ)

- **`code/result/niches.json`** — масив об’єктів: `niche_name`, `niche_description`, `competitors`, `metadata`. Можлива ніша «Unclassified / Unique Apps».
- **`code/result/app_niche_mapping.csv`** — колонки `AppName`, `Description`, `SubNicheName`; кодування `utf-8-sig`.

### 3.2 Метрики останнього LLM-прогону

Джерело: `code/result/metrics.json`.

| Метрика | Значення |
| --- | --- |
| Кластерів (`n_clusters`, без noise) | 468 |
| Шум (`n_noise`) | 1 |
| Silhouette (cosine, inliers) | 0.1783 |
| Середня внутрішньокластерна схожість | 0.6818 |
| Медіана розміру кластера | 7 |
| min / max / mean розміру | 3 / 81 / 8.9 |

Метрики **допоміжні**; остаточна якість — вибірковий перегляд ніш і UMAP.

### 3.3 UMAP

`code/result/umap_clusters.html` — 2D UMAP, Plotly, підказки з `trackName`.

### 3.4 Прогін без LLM

`python segment.py --no-llm` пише в **`code/result/` ті самі імена файлів**, що й повний прогін — **окремого шляху в коді немає**. Щоб зберегти **обидві** версії локально, копіюють результати, наприклад у `code/result/no_llm/` з іменами на кшталт `niches_no_llm.json` (конвенція; каталог часто в `.gitignore`). Продуктова різниця: з LLM зазвичай **кращі назви та описи ніш**.

### 3.5 Приклади ніш з `niches.json`

- Photo Typography Editors  
- Property Boundary Mapping Apps  
- Connected Home Fitness Apps  

(наближено до прикладів гранулярності з ТЗ).

### 3.6 Обмеження

Embedding-кластеризація не гарантує 100% прямих конкурентів у кожній клітинці; silhouette на реальних ASO-текстах часто помірний; рекомендований **spot-check** 10–20 ніш.

### 3.7 Опційний refinement (`refine_niches.py`)

Окремий скрипт **після** `segment.py`: читає `niches.json` і `cache/embeddings.npy`, LLM-валідація кожної ніші (outliers), перепризначення за косинусом до центроїдів інших ніш (поріг `REFINE_REASSIGN_THRESHOLD`, типово ~0.58) або в **`Refinement / Unassigned`**. Вихід — **нові** файли з суфіксом `_refined` (без перезапису базових). Плюси: вища середня внутрішньокластерна схожість на тих самих ембеддингах (~0.68→~0.76 у прогоні), розведення близьких підкатегорій, логи `[1/5]–[5/5]` і прогрес LLM. Ризики: може знизитися silhouette; частина додатків у корзині unassigned; рідкі семантично спірні переноси лише за вектором. **Подальшу розробку цього етапу не плануємо** — зафіксовано як додатковий QA-шар.

Артефакти: `niches_refined.json`, `app_niche_mapping_refined.csv`, `metrics_refined.json`, `umap_clusters_refined.html`. Запуск: `cd code` → `python refine_niches.py`.

---

## 4. Резюме

Реалізовано пайплайн сегментації ~4200 додатків: підготовка тексту, опційне LLM-збагачення, ембеддинги (локально за замовчуванням або ланцюг API з fallback на bge), agglomerative кластеризація з постобробкою, LLM-неймінг з fallback, метрики, JSON/CSV/HTML. Опційно — **`refine_niches.py`** і артефакти `*_refined`. Конфігурація через `.env`.

---

## 5. Де що лежить і як запустити

| Що | Шлях |
| --- | --- |
| Звіт | `doc/Звіт_тестове_ML_Engineer_genAI.md` |
| ТЗ | `Тестове завдання на ML Engineer genAI (LLM).md` |
| Код | `code/segment.py`, `code/config.py`, за потреби `code/refine_niches.py` |
| Результати основного прогону | `code/result/niches.json`, `app_niche_mapping.csv`, `metrics.json`, `umap_clusters.html`, кеш у `code/result/cache/` |
| Refinement | `code/result/*_refined*` |

**Команди:** `python segment.py` — повний цикл; `python segment.py --no-llm` — без LLM; `python refine_niches.py` — після основного прогону (потрібні ті самі LLM-ключі, що для сегментації).

---

*Кінець звіту.*
