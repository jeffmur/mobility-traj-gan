import click
import pandas as pd
import sys

print(sys.path)
from src.models import marc


@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.argument("result_file", type=click.Path())
@click.argument("epochs", type=click.INT)
def train_marc(train_file, test_file, result_file, epochs):
    """Train the MARC trajectory-user linking classifier."""

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    marc.train(train_df, test_df, epochs)


@click.group()
def cli():
    """Command line interface for the mobility learning framework."""


def main():
    cli.add_command(train_marc)
    cli()


if __name__ == "__main__":
    main()
