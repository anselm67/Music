#!/usr/bin/env python3
import logging

import click
from torchinfo import summary as model_summary

from models import Config, HierarchicalDETR


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
def cli(log_level: str):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@click.command()
def summary():
    config = Config()
    model = HierarchicalDETR(config)
    model_summary(model, input_size=(config.batch_size,
                  config.in_channels, *config.image_shape))


cli.add_command(summary)


def main():
    cli()


if __name__ == '__main__':
    main()
