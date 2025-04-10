import os
import sqlite3

from utils import get_logger

logger = get_logger()

db_name = os.environ.get("SQL_LITE_DB_NAME", "tracking")
table_name = os.environ.get("SQL_LITE_TABLE_NAME", "events")


def run_sql(statement: str):
    try:
        with sqlite3.connect(f"data/{db_name}.db") as conn:
            # create a cursor
            cursor = conn.cursor()

            # execute statements
            cursor.execute(statement)

            # commit the changes
            conn.commit()
    except sqlite3.OperationalError as e:
        logger.info("Failed to run code:", e)


def create_table():
    table_statement = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            detection text NOT NULL,
            event_date DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now', 'localtime'))
        );"""
    run_sql(table_statement)


def insert_one(prediction: str):
    statement = f"""INSERT INTO events ('detection') VALUES ('{prediction}');"""
    run_sql(statement)


if __name__ == "__main__":
    create_table()
