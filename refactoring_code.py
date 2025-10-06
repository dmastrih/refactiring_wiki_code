import time
import requests
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import logging
import sys


API_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics"
TOP_ENDPOINT = "pageviews/top"
DEFAULT_PROJECT = "en.wikipedia"
DEFAULT_ACCESS = "all-access"
MAX_RETRIES = 3
REQ_TIMEOUT = 30
TOP_ARTICLES = 20
OUTPUT_FILENAME = "img/ref_top_articles.png"


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logger


def get_top_wiki_articles(project, year, month, day, access=DEFAULT_ACCESS):
    """Получение топовых статей"""

    top_args = f"{project}/{access}/{year}/{month}/{day}"
    url = "/".join([API_BASE_URL, TOP_ENDPOINT, top_args])

    for retr in range(MAX_RETRIES):
        try:
            logger.info(
                f"Запрос к API за {year}-{month}-{day} "
                f"(попытка {retr + 1})"
            )
            response = requests.get(
                url, headers={"User-Agent": "wiki parser"},
                timeout=REQ_TIMEOUT
            )
            # В документации пишут, что можно отправлять 200 запросов в час
            if response.status_code == 429:
                wait_time = 2 ** retr + 1
                logger.warning(
                    f"Лимит запросов превышен. Ждем {wait_time} секунд"
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Ошибка запроса (попытка {retr + 1}): {e}"
            )
            if retr < MAX_RETRIES - 1:
                time.sleep(1)

    logger.error(f"Не удалось получить данные за {year}-{month}-{day}")
    return None


def validate_dates(start_date, end_date):
    """Валидация диапазона дат"""

    if start_date > end_date:
        raise ValueError("Начальная дата больше конечной")

    if start_date > dt.datetime.now():
        raise ValueError("Начальная дата находится в будущем")

    logger.info("Валидация дат завершена")


def collect_wiki_data(start_date, end_date):
    """Сбор данных из Википедии за указанный период"""

    delta = dt.timedelta(days=1)
    current_date = start_date

    df = pd.DataFrame()
    while current_date <= end_date:
        year = str(current_date.year)
        month = str(current_date.month)
        day = str(current_date.day)
        current_date += delta

        data = get_top_wiki_articles(DEFAULT_PROJECT, year, month, day)

        if data and "items" in data and len(data["items"]) > 0:
            articles_data = pd.DataFrame(data["items"][0]["articles"])
            articles_data["date"] = f"{year}{month}{day}"
            df = pd.concat([df, articles_data])
            logger.info(
                f"Получено {len(articles_data)} статей за {year}-{month}-{day}"
            )
        else:
            logger.warning(f"Нет данных за {year}-{month}-{day}")

    if df.empty:
        raise ValueError("Не удалось получить данные")

    logger.info(f"Собрано {len(df)} записей")

    return df


def process_wiki_data(df):
    """Обработка данных"""

    df["date"] = pd.to_datetime(df["date"])
    idx = pd.MultiIndex.from_product(
        [df["article"].unique(),
         pd.date_range(start=df["date"].min(), end=df["date"].max())],
        names=["article", "date"]
    )

    df.set_index(["article", "date"], inplace=True)
    df = df.reindex(idx)
    df = df.reset_index(drop=False)

    df["views"] = df.groupby("article")["views"].transform(lambda x: x.ffill())
    last_values = df.groupby("article")["views"].last()
    top_articles = last_values.nlargest(TOP_ARTICLES)
    df_top_articles = df[df["article"].isin(top_articles.index)]

    return df_top_articles


def calculate_statistics(df, df_top_articles):
    """Расчет статистики"""

    views_sum = {article: 0 for article in df_top_articles["article"].unique()}
    count = {article: 0 for article in df_top_articles["article"].unique()}

    for i in range(len(df_top_articles)):
        article = df_top_articles.iloc[i]["article"]
        views = df_top_articles.iloc[i]["views"]
        views_sum[article] += views
        count[article] += 1

    mean_views = {article: views_sum[article] / count[article] for article in views_sum}
    mean_views = int(np.nanmean(list(mean_views.values())))
    max_views = df["views"].max()
    unique_articles = df["article"].nunique()

    return mean_views, max_views, unique_articles


def create_plot(df_top_articles, mean_views, max_views, unique_articles):
    """Создание графика просмотров"""

    logger.info("Создание графика")

    title = (f"Top articles wiki views "
             f"(Mean: {mean_views:.2f}, Max: {max_views}, "
             f"Articles: {unique_articles})")

    plt.figure(figsize=(12, 8))
    for article in df_top_articles["article"].unique():
        df_article = df_top_articles[df_top_articles["article"] == article]
        plt.plot(df_article["date"], df_article["views"], label=article)

    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.savefig(OUTPUT_FILENAME)
    logger.info(f"График сохранен в '{OUTPUT_FILENAME}'")


def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Process start and end dates.")
    parser.add_argument("start", type=str, help="The start date in YYYYMMDD format")
    parser.add_argument("end", type=str, help="The end date in YYYYMMDD format")
    args = parser.parse_args()

    start = args.start
    end = args.end

    try:
        start_date = dt.datetime.strptime(start, "%Y%m%d")
        end_date = dt.datetime.strptime(end, "%Y%m%d")

        validate_dates(start_date, end_date)

        df = collect_wiki_data(start_date, end_date)
        df_original = df.copy()

        df_top_articles = process_wiki_data(df)

        mean_views, max_views, unique_articles = calculate_statistics(df_original, df_top_articles)

        create_plot(df_top_articles, mean_views, max_views, unique_articles)

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
