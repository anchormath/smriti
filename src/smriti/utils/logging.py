"""
Console logging and formatting utilities using Rich.
"""

from rich.console import Console
from rich.table import Table

console = Console()


def print_header(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def print_metric_table(metrics: dict, title: str = "Metrics Summary") -> None:
    table = Table(title=title)
    table.add_column("Property", style="magenta")
    table.add_column("Value", style="green")

    for k, v in metrics.items():
        table.add_row(str(k), str(v))

    console.print(table)
