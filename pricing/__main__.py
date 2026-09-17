"""CLI: train models from crawl DB."""
from __future__ import annotations

import click

from domain.model_key import SUPPORTED_CITIES, SUPPORTED_PURPOSES, ModelKey
from training.pipeline import train_model


@click.group()
def cli() -> None:
    """Didibood Price — train CatBoost models."""


@cli.command("train")
@click.option("--city", "-c", required=True, type=click.Choice(SUPPORTED_CITIES))
@click.option("--property-type", default="apartment", show_default=True)
@click.option(
    "--purpose",
    default="sale",
    show_default=True,
    type=click.Choice(SUPPORTED_PURPOSES),
)
@click.option("--refresh-data", is_flag=True, help="Re-extract from PostgreSQL.")
def train_cmd(city: str, property_type: str, purpose: str, refresh_data: bool) -> None:
    """Train one model (sale or rent)."""
    key = ModelKey(city_slug=city, property_type=property_type, purpose=purpose)
    result = train_model(key, refresh_data=refresh_data)
    click.echo(f"Trained {result.model_key}")
    click.echo(f"  rows={result.n_rows} train={result.n_train} test={result.n_test}")
    click.echo(f"  MAE={result.mae:,.0f} MAPE={result.mape:.2%}")
    click.echo(f"  saved={result.model_path}")


@cli.command("refresh-analytics")
@click.option("--city", "-c", default=None, type=click.Choice(SUPPORTED_CITIES))
@click.option("--property-type", default="apartment", show_default=True)
def refresh_analytics_cmd(city: str | None, property_type: str) -> None:
    """Rebuild the parquet the neighbourhood analytics endpoints read.

    Run it after a crawl: the API serves whatever this last wrote, and only
    rebuilds on its own when the file is missing or has gone stale.
    """
    from application.neighbourhoods.dataset import refresh_analytics_frame

    cities = [city] if city else list(SUPPORTED_CITIES)
    for city_slug in cities:
        for purpose in SUPPORTED_PURPOSES:
            key = ModelKey(city_slug=city_slug, property_type=property_type, purpose=purpose)
            try:
                path = refresh_analytics_frame(key)
                click.echo(f"{key.slug()} -> {path}")
            except Exception as exc:  # one bad key must not stop the rest
                click.echo(f"{key.slug()} FAILED: {exc}", err=True)


@cli.command("train-all")
@click.option("--refresh-data", is_flag=True)
@click.option(
    "--purpose",
    default="sale",
    show_default=True,
    type=click.Choice(SUPPORTED_PURPOSES),
)
def train_all(refresh_data: bool, purpose: str) -> None:
    """Train apartment models for tehran, mashhad, isfahan."""
    for city in SUPPORTED_CITIES:
        key = ModelKey(city_slug=city, property_type="apartment", purpose=purpose)
        click.echo(f"=== {key.slug()} ===")
        try:
            result = train_model(key, refresh_data=refresh_data)
            click.echo(
                f"OK rows={result.n_rows} MAE={result.mae:,.0f} "
                f"MAPE={result.mape:.2%}"
            )
        except ValueError as exc:
            click.echo(f"SKIP: {exc}")


if __name__ == "__main__":
    cli()
