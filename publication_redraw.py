"""Redraw two revised publication figures from included aggregate tables."""
from pathlib import Path
import argparse
import importlib
import json
import shutil
import sys
import types


ROOT = Path(__file__).resolve().parent


def plotting_module():
    """Load plotting helpers without importing the full large-data runtime."""
    runtime = types.ModuleType('assembly.runtime')
    runtime.ROOT = ROOT

    def dirs(settings):
        root = Path(settings['output_dir'])
        result = {'root': root}
        for name in ['cache', 'data', 'tables', 'figures', 'r', 'status']:
            result[name] = root / name
            result[name].mkdir(parents=True, exist_ok=True)
        return result

    def csv(frame, path):
        frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.12g')

    def json_write(value, path):
        Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')

    runtime.dirs = dirs
    runtime.csv = csv
    runtime.json_write = json_write
    previous = sys.modules.get('assembly.runtime')
    sys.modules['assembly.runtime'] = runtime
    try:
        return importlib.import_module('assembly.figures')
    finally:
        if previous is None:
            sys.modules.pop('assembly.runtime', None)
        else:
            sys.modules['assembly.runtime'] = previous


def redraw(tables_dir, output_dir):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    tables_dir = Path(tables_dir).resolve()
    output_dir = Path(output_dir).resolve()
    names = ['fields', 'countries', 'population_year', 'role_country_period', 'paper_annual']
    required = [tables_dir / f'{name}.csv' for name in names]
    required.append(tables_dir / 'Regression_Coefficients.csv')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing aggregate tables:\n' + '\n'.join(missing))
    figures = plotting_module()
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    for name in names:
        source = tables_dir / f'{name}.csv'
        destination = output_dir / 'tables' / source.name
        if source != destination:
            shutil.copyfile(source, destination)
    settings = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    settings.update(output_dir=str(output_dir), _project_dir=str(ROOT))

    class CountrySafeData(figures.Data):
        def read(self, name):
            if name == 'countries':
                return pd.read_csv(self.d['tables'] / f'{name}.csv', keep_default_na=False)
            return super().read(name)

        def save(self, figure, name, caption):
            for axis in figure.axes:
                if axis.get_xlabel() == 'Bridge credit / exposure, relative to world':
                    axis.set_xlabel('Contextual credit / exposure, relative to world', fontsize=5.8)
            super().save(figure, name, caption)

    figures.figure1(CountrySafeData(settings))
    # Preserve the four reported specifications, estimates and confidence intervals.
    coefficients = pd.read_csv(tables_dir / 'Regression_Coefficients.csv')
    coefficients = coefficients[coefficients.model != 'paper_AI_profile']
    coefficients.to_csv(output_dir / 'tables' / 'Regression_Coefficients.csv', index=False)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'axes.titlesize': 9})
    figure, axes = plt.subplots(2, 2, figsize=(7.3, 4.75), layout='constrained')
    labels = {'international': 'International authorship', 'log_team': 'Log author count',
              'log_refs': 'Log usable references', 'mean_ref_age_joint': 'Mean reference age (years)',
              'log1p(RCA)': 'Log(1 + RCA)', 'log1p(E_asm)': 'Log(1 + assembler exposure)'}
    models = [('paper_joint_cross', 'Joint co-use, q'), ('paper_geo_cross', 'Geographic co-use, g'),
              ('paper_field_cross', 'Disciplinary co-use, d'), ('country_field_assembly', 'Assembler intensity')]
    for index, (axis, (model, title)) in enumerate(zip(axes.flat, models)):
        rows = coefficients[coefficients.model == model]
        y = np.arange(len(rows))
        axis.axvline(0, color='#9AA3A9', lw=.7, zorder=0)
        x = rows.estimate.to_numpy()
        errors = np.array([x - rows.conf_low.to_numpy(), rows.conf_high.to_numpy() - x])
        axis.errorbar(x, y, xerr=errors, fmt='o', markersize=4,
                      color=figures.TEAL if index < 3 else figures.NAVY,
                      capsize=2, elinewidth=1)
        axis.set_yticks(y, [labels[term] for term in rows.term])
        axis.invert_yaxis()
        axis.set_ylim(len(rows) - .6, -.55)
        axis.set_title(chr(97 + index), loc='left', fontweight='bold')
        axis.set_title(title, loc='center', fontweight='normal')
        axis.set_xlabel('Coefficient (95% confidence interval)')
        axis.ticklabel_format(axis='x', style='plain', useOffset=False)
        axis.margins(x=.15)
        axis.grid(axis='x', alpha=.12)
    for extension in ['pdf', 'svg', 'png']:
        figure.savefig(output_dir / 'figures' / f'Supplementary_Figure_07_Conditional_Associations.{extension}', dpi=400)
    plt.close(figure)
    print('Redrawn two figures from existing numerical results; retained 4 models and '
          f'{len(coefficients)} coefficients.')
    print('Output:', output_dir / 'figures')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', type=Path, default=ROOT / 'publication_tables')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'publication_output')
    arguments = parser.parse_args(argv)
    redraw(arguments.tables_dir, arguments.output_dir)


if __name__ == '__main__':
    main()
