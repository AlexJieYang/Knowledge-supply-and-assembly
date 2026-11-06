"""Build SI tables, detailed figures, Source Data and a combined PDF."""
from pathlib import Path
import argparse
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=os.environ.get('GKA_CONFIG',str(Path(__file__).with_name('config.json'))))
    parser.add_argument('--results-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--sample-file',help='Optional CSV/Parquet sample; custom inclusion design is explicitly labelled.')
    parser.add_argument('--min-exposure',type=float)
    parser.add_argument('--top-countries',type=int,default=24)
    parser.add_argument('--stage',choices=['all','tables','figures'],default='all')
    parser.add_argument('--population-diagnostics',dest='population_diagnostics',action='store_true',help='Stream existing assembly_papers.parquet for population diagnostics.')
    parser.add_argument('--batch-rows',type=int,default=250000)
    parser.add_argument('--no-book',action='store_true')
    parser.add_argument('--synthetic',action='store_true',help='Watermark every output as synthetic; never use for empirical results.')
    args=parser.parse_args()
    if args.batch_rows<1:parser.error('--batch-rows must be positive')
    for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    from si_support.core import Context,MissingInput
    ctx=Context(args.config,args.results_dir,args.output_dir,args.sample_file,args.min_exposure,args.top_countries,args.synthetic)
    if args.stage in ['all','tables']:
        from si_support.analysis import build_tables,population_diagnostics
        build_tables(ctx)
        if args.population_diagnostics:
            try:population_diagnostics(ctx,args.batch_rows)
            except MissingInput as exc:ctx.event('tables','population_diagnostics','skipped',exc)
            else:ctx.event('tables','population_diagnostics','complete')
    if args.stage in ['all','figures']:
        from si_support.figures import render_all
        render_all(ctx,book=not args.no_book)
    ctx.finish()
    print('SI output:',ctx.out)
    print('Read SI_run_status.csv for completed and skipped sections.')


if __name__=='__main__':main()
