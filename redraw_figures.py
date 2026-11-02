"""Redraw the five publication figures from completed result tables/caches only."""
from pathlib import Path
import argparse
import json


TABLES = (
    'fields', 'countries', 'population_year', 'role_country_period', 'paper_annual',
    'community_runs', 'community_stability', 'network_layout', 'alluvial_countries',
    'community_similarity', 'network_node_roles', 'histograms', 'field_year_means',
    'response_surface', 'role_country_field_period', 'role_lorenz',
    'fixed_exposure_clubs', 'change_accounting', 'cohort_intensity', 'network_annual',
    'definition_sensitivity', 'fixed_support_mean', 'joint_semantic',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--results-dir', help='Optional existing results directory; overrides config.output_dir.')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    settings = json.loads(path.read_text(encoding='utf-8-sig'))
    settings['_project_dir'] = str(path.parent)
    result = Path(args.results_dir).resolve() if args.results_dir else Path(settings['output_dir'])
    if not result.is_absolute():
        result = path.parent / result
    settings['output_dir'] = str(result)
    required = [result/'tables'/(name+'.csv') for name in TABLES]
    required += [result/'cache/window_networks.npz', result/'data/plot_sample.parquet']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            'Completed figure inputs are missing. Point --results-dir to the results from stages 01-05.\n'
            + '\n'.join(missing)
        )
    import bootstrap
    bootstrap.configure(settings.get('threads', 8))
    from assembly.figures import build
    summary = build(settings)
    print(f'Redrawn {summary["figures"]} figures: {result / "figures"}')
    print('Figure titles, panel descriptions and notes are saved separately in Fig*_caption.txt.')


if __name__ == '__main__':
    main()
