import numpy as np
import pandas as pd
import pytest
from assembly.runtime import ids,Lookup,Writer
from assembly.metadata import resolve
from assembly.networks import balanced_groups


def test_exact_ids_above_double_precision():
    v=np.array([2**53+1,2**53+3],dtype=np.uint64)
    np.testing.assert_array_equal(ids(pd.Series(v)),v)
    np.testing.assert_array_equal(ids(pd.Series(['W9007199254740993','https://openalex.org/W9007199254740995'])),v)
    with pytest.raises(ValueError):ids(pd.Series([float(2**53+1)]))
    with pytest.raises(ValueError):ids(pd.Series([-1,2]))
    for mode in ('pandas','sorted'):
        lookup=Lookup(v,mode);np.testing.assert_array_equal(lookup.get([v[1],np.uint64(7),v[0]]),[1,-1,0])


def test_writer_atomic_failure_and_row_groups(tmp_path):
    p=tmp_path/'one.parquet'
    with Writer(p) as w:
        w.write(pd.DataFrame({'a':np.arange(3,dtype=np.int64)}))
        w.write(pd.DataFrame({'a':np.arange(3,8,dtype=np.int64)}))
    assert pd.read_parquet(p).a.tolist()==list(range(8))
    with pytest.raises(RuntimeError):
        with Writer(p) as w:
            w.write(pd.DataFrame({'a':[99]}));raise RuntimeError('simulated interrupted writer')
    assert pd.read_parquet(p).a.tolist()==list(range(8))
    assert not p.with_name(p.name+'.building').exists()


def test_adapter_does_not_silently_choose_ambiguous_fields():
    with pytest.raises(ValueError):resolve(['field_id','category_id'],'auto',['field_id','category_id'])
    assert resolve(['category_id'],'auto',['field_id','category_id'])=='category_id'


def test_year_scheduling_preserves_all_paper_groups():
    gp=np.array([0,2,3,7,9],np.int64);gr=np.arange(9,dtype=np.uint32);counts=np.array([2,3,5,20,30,40,50,60,70])
    p,r,order=balanced_groups(gp,gr,counts,2)
    np.testing.assert_array_equal(np.sort(order),np.arange(4))
    np.testing.assert_array_equal(np.sort(r),gr)
    for j,y in enumerate(order):np.testing.assert_array_equal(r[p[j]:p[j+1]],gr[gp[y]:gp[y+1]])
