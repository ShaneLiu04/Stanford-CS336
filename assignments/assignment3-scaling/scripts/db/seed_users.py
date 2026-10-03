import csv
import os
from pathlib import Path
from typing import Annotated

os.environ["DB_ENV"] = "prod"

import typer
from sqlalchemy import select
from sqlalchemy.orm import Session

from cs336_scaling.config import settings_from_env
from cs336_scaling.db import Base, get_engine, get_session_factory, init_db
from cs336_scaling.db.tables import UserTable

DEFAULT_CSV_PATH = Path("seeds/students.csv")


def main(
    csv_path: Annotated[
        Path,
        typer.Argument(
            help="Path to a CSV file whose first row is exactly sunet_id,api_key.",
        ),
    ] = DEFAULT_CSV_PATH,
) -> None:
    inserted_count = seed_users_from_csv(csv_path)
    typer.echo(f"Seeded {inserted_count} users from {csv_path}.")


def seed_users_from_csv(csv_path: str | Path) -> int:
    configure_production_database()
    init_db()
    session_factory = get_session_factory()

    parsed_rows = parse_users_csv(csv_path)

    with session_factory() as session, session.begin():
        if not database_is_empty(session):
            raise ValueError("refusing to seed a non-empty database")

        session.add_all(
            UserTable(sunet_id=sunet_id, api_key=api_key)
            for sunet_id, api_key in parsed_rows
        )

    return len(parsed_rows)


def configure_production_database() -> None:
    os.environ["DB_ENV"] = "prod"
    settings_from_env.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


def database_is_empty(session: Session) -> bool:
    return all(
        session.execute(select(next(iter(table.c))).select_from(table).limit(1)).first()
        is None
        for table in Base.metadata.sorted_tables
    )


def parse_users_csv(csv_path: str | Path) -> list[tuple[str, str]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header != ["sunet_id", "api_key"]:
            raise ValueError("expected first row to be exactly 'sunet_id,api_key'")

        return [(sunet_id.strip(), api_key.strip()) for sunet_id, api_key in reader]


if __name__ == "__main__":
    typer.run(main)
