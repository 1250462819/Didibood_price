# Didibood Price

سرویس پیش‌بینی قیمت ملک با **CatBoost** — مدل جدا per **شهر + apartment + sale/rent**.

## Local (Mac)

```bash
cd Didibood_price
cp .env.local.example .env   # optional — run-local creates it
./scripts/run-local.sh
# Swagger: http://127.0.0.1:8093/docs
```

- **Predict** بدون DB کار می‌کند اگر مدل‌ها در `artifacts/models/*.cbm` باشند.
- **Train** با `refresh-data` به `DATABASE_URL` نیاز دارد — روی سرور راحت‌تر است (DB مستقیم روی `:5432`).

```bash
# train روی سرور (پیشنهادی)
ssh ubuntu@SERVER 'cd /opt/didibood/Didibood_Price && .venv/bin/python -m pricing train-all --refresh-data'
ssh ubuntu@SERVER 'cd /opt/didibood/Didibood_Price && .venv/bin/python -m pricing train-all --purpose rent --refresh-data'
```

## Production deploy (rsync + systemd)

```bash
export DEPLOY_HOST=37.32.12.208
export DEPLOY_USER=ubuntu
LOCAL_DEPLOY=true ./scripts/deploy-production.sh
```

مسیر سرور: `/opt/didibood/Didibood_Price` — سرویس `didibood-price` روی `:8093`.

اولین deploy: اگر `.env` نباشد از `.env.production.example` ساخته می‌شود؛ `DATABASE_URL` از `.env` کراولر Divar کپی می‌شود (در صورت وجود).

## Environment files

| File | Use |
|------|-----|
| `.env.local.example` | توسعه روی Mac |
| `.env.production.example` | سرور production |
| `.env.example` | راهنمای کوتاه |

## Train

```bash
python -m pricing train -c tehran --refresh-data
python -m pricing train-all --purpose rent --refresh-data
```

مدل‌ها: `artifacts/models/{city}__apartment__{sale|rent}.cbm`

## API

| | Endpoint |
|---|----------|
| پیش‌بینی فروش | `POST /api/v1/sale/predict` |
| train فروش | `POST /api/v1/sale/train` |
| پیش‌بینی اجاره | `POST /api/v1/rent/predict` |
| train اجاره | `POST /api/v1/rent/train` |

### Sale predict

```json
{
  "city_slug": "tehran",
  "property_type": "apartment",
  "neighbourhood": "جردن",
  "area": 85,
  "rooms": 2
}
```

### Rent predict

```json
{
  "city_slug": "tehran",
  "area": 85,
  "neighbourhood": "جردن",
  "rooms": 2
}
```

**Rent Y:** `equivalent_deposit_toman` = `deposit + monthly × (10M / 300K)`

## Neighbourhood analytics (the `/neighborhoods` page)

دو لایهٔ کش، هر دو شبانه:

| لایه | چیست | کجا |
|------|------|-----|
| frame | لیستینگ‌های تمیزشدهٔ هر شهر | `artifacts/datasets/analytics__*.parquet` |
| answers | خود پاسخ اندپوینت‌ها، از پیش حساب‌شده | `artifacts/datasets/answers/*.json.gz` |

یک یونیت systemd هر شب هر دو را می‌سازد (`didibood-price-analytics.timer`):

```bash
.venv/bin/python -m pricing refresh-analytics      # frames از کرال
.venv/bin/python -m pricing precompute-answers     # پاسخ‌های صفحهٔ محله‌ها
```

`precompute-answers` دقیقاً همان چیزی را حساب می‌کند که صفحه می‌پرسد: ۸۰ حالت
فیلتر (۴ امکانات × ۵ گزینهٔ اتاق) برای هر شهر، گزارش هر محله، و فهرست شهرها.
درخواستی که پاسخ آماده دارد فقط یک فایل می‌خواند؛ هر فیلتر دیگری مثل قبل زنده
حساب می‌شود. پاسخ کهنه‌تر از `NEIGHBOURHOOD_ANSWER_MAX_AGE_HOURS` (پیش‌فرض ۳۶
ساعت) سرو نمی‌شود — اگر جاب شبانه بخوابد، صفحه کند می‌شود نه غلط.

## Features (Didibood-aligned)

`neighbourhood`, `area`, `rooms`, `year_built`, `floor_number`, amenities, `location_lat/long`

## v1 scope

- شهرها: tehran, mashhad, isfahan
- فروش: `price_per_sqm_toman`
- اجاره: `equivalent_deposit_toman`
