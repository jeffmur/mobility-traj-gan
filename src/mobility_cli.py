import click
from src.models import marc


@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.argument("result_file", type=click.Path())
@click.argument("dataset_name", type=click.STRING)
def train_marc():
    """Train the MARC trajectory-user linking classifier."""
    marc.train()


@click.group()
def cli():
    """Command line interface for the mobility learning framework."""


def main():
    cli.add_command()
    cli()


if __name__ == "__main__":
    main()
