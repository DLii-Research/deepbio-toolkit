import argparse
from pathlib import Path
from typing_extensions import override
from .._cli import CliSubCommand, subcommand

__doc__ = "Database tools for Deepbio Toolkit"

@subcommand("import_sequences", "Generate a random run ID.")
class ImportSequences(CliSubCommand):
    @override
    def configure(self, parser):
        parser.add_argument("input_path", type=Path, help="Input FASTA or FASTQ file containing DNA sequences.")
        parser.add_argument("--output", "-O", type=Path, required=False, help="Path to the output .seq file.")
        parser.add_argument("--silent", "-s", action="store_true", help="Do not show progress bar.")

    @override
    def run(self, config: argparse.Namespace) -> int:
        # Input processing
        suffix = config.input_path.name.rstrip(".gz")
        if suffix.endswith(".fastq"):
            from dnadb import fastq
            entries = fastq.entries(config.input_path)
        elif suffix.endswith(".fasta"):
            from dnadb import fasta
            entries = fasta.entries(config.input_path)
        else:
            print("Unknown file type: ", suffix)
            return 1
        # Output processing
        output_path = config.output
        if output_path is None:
            name = config.input_path.name.rstrip(".gz")
            output_path = config.input_path.with_name(name)
            if not output_path.name.endswith(".seq"):
                output_path = output_path.with_suffix(".seq")
        # Write sequences
        from ...data.databases import SequenceDb
        sequences = [e.sequence for e in entries]
        SequenceDb.create(output_path, sequences, progress=not config.silent)
        if not config.silent:
            print("Done. Imported", len(sequences), "sequences(s).")