"""CLI commands to plot the results."""

import typer

from .eigenfrequency_spectrums_both import app as eigenfreq_spec_both_app
from .eigenfrequency_spectrums_by_length import app as eigenfreq_spec_by_len_app
from .eigenfrequency_spectrums_by_n import app as eigenfreq_spec_by_n_app
from .eigenmodes import app as eigenmods_app
from .laplacian import app as laplacian_app
from .measure_relative_error_sparse_eigenvalues import (
    app as measure_rel_err_spars_eigval_app,
)
from .measure_sparse_dense_eig_runtime import (
    app as measure_sparse_dense_eig_runtime_app,
)

app = typer.Typer(no_args_is_help=True)

app.add_typer(measure_rel_err_spars_eigval_app)
app.add_typer(measure_sparse_dense_eig_runtime_app)
app.add_typer(eigenfreq_spec_by_len_app)
app.add_typer(eigenfreq_spec_by_n_app)
app.add_typer(eigenfreq_spec_both_app)
app.add_typer(eigenmods_app)
app.add_typer(laplacian_app)
