import numpy as np
from assembly.kernels import metric_batch,sort_unique_rows,build_pools,aggregate_networks
from assembly.networks import partition_similarity


def csr(rows):
    ptr=np.r_[0,np.cumsum([len(x) for x in rows])].astype(np.int64)
    return ptr,np.array([x for row in rows for x in row],np.int16)


def test_reference_pair_formula_and_self_exclusion():
    rng=np.random.default_rng(930)
    for trial in range(40):
        m=int(rng.integers(2,17));nc=4;nf=3
        a=rng.dirichlet(np.ones(nc),size=m);t=rng.dirichlet(np.ones(nf),size=m)
        cp,cc=csr([[]]+[list(range(nc)) for _ in range(m)])
        fp,fc=csr([[]]+[list(range(nf)) for _ in range(m)])
        cw=a.ravel();fw=t.ravel();rp=np.r_[0,m,np.repeat(m,m)].astype(np.int64)
        rn=np.r_[m,np.zeros(m)].astype(np.uint32);ri=np.arange(1,m+1,dtype=np.uint32)
        year=np.repeat(2020,m+1).astype(np.int16);pri=np.r_[-1,np.argmax(t,axis=1)].astype(np.int16)
        sigma=np.full(m,np.nan,np.float32)
        z=metric_batch(0,1,1,nc,nf,m,year,pri,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma)
        g=1-a@a.T;f=1-t@t.T;r=g*f
        np.fill_diagonal(g,0);np.fill_diagonal(f,0);np.fill_diagonal(r,0)
        brute=r.sum(1)/(m-1)
        np.testing.assert_allclose(sigma,brute,atol=1e-7)
        np.testing.assert_allclose(z[0,1:4],[g.sum()/m/(m-1),f.sum()/m/(m-1),brute.mean()],atol=1e-7)
        # Ignoring the diagonal is essential with fractional country/field memberships.
        np.testing.assert_allclose(np.nansum(sigma)/m,z[0,3],atol=1e-7)


def test_exact_three_reference_example():
    cp,cc=csr([[1],[0],[0],[1]]);fp,fc=csr([[1],[0],[0],[1]])
    cw=np.ones(4);fw=np.ones(4);rp=np.array([0,3,3,3,3],np.int64);rn=np.array([3,0,0,0],np.uint32);ri=np.array([1,2,3],np.uint32)
    s=np.zeros(3,np.float32);year=np.array([2020,2000,2000,2010],np.int16);primary=np.array([1,0,0,1],np.int16)
    z=metric_batch(0,4,2,2,2,3,year,primary,cp,cc,cw,fp,fc,fw,rp,rn,ri,s)
    np.testing.assert_allclose(s,[.5,.5,1]);np.testing.assert_allclose(z[0,3],2/3)
    assert np.isnan(z[1,3])
    # All main references measurable: completion bounds collapse to q.
    np.testing.assert_allclose(z[0,6:8],[2/3,2/3])


def test_sort_deduplicate_across_input_order():
    p=np.array([0,5,5,8],np.int64);a=np.array([3,1,3,2,1,7,6,7],np.uint32);n=np.empty(3,np.uint32)
    sort_unique_rows(p,a,n)
    assert n.tolist()==[3,0,2]
    assert a[:3].tolist()==[1,2,3] and a[5:7].tolist()==[6,7]


def test_ami_label_invariance_and_unassigned():
    a=np.array([0,0,1,1,-1]);b=np.array([7,7,2,2,6])
    ami,vi,n=partition_similarity(a,b)
    np.testing.assert_allclose([ami,vi],[1,0],atol=1e-12);assert n==4
    assert partition_similarity(np.zeros(4,int),np.ones(4,int))[0]==1


def test_label_pool_against_enumerated_leave_paper_out_partners():
    # Every focal paper has fractional receiving Fields; the focal paper's ENTIRE
    # set of reference slots must be removed from each conditional pool.
    rng=np.random.default_rng(77);n=9;nc=3;nf=2
    cp,cc=csr([list(range(nc)) for _ in range(n)]);fp,fc=csr([list(range(nf)) for _ in range(n)])
    a=rng.dirichlet(np.ones(nc),n);t=rng.dirichlet(np.ones(nf),n);cw=a.ravel();fw=t.ravel()
    refsets=[np.array([(i+1)%n,(i+2)%n,(i+4)%n],np.uint32) for i in range(n)]
    rp=np.arange(0,3*(n+1),3,dtype=np.int64);rn=np.full(n,3,np.uint32);ri=np.concatenate(refsets)
    year=np.full(n,2020,np.int16);kind=np.zeros(n,np.int8);pri=np.argmax(t,axis=1).astype(np.int16);sigma=np.full(len(ri),np.nan,np.float32)
    z=metric_batch(0,n,2,nc,nf,3,year,pri,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma)
    gp=np.array([0,n],np.int64);gr=np.arange(n,dtype=np.uint32)
    pools=build_pools(gp,gr,nc,nf,year,kind,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma)
    qzero=np.full(n,np.nan,np.float32)
    net,role,cohort,nmodel=aggregate_networks(gp,gr,nc,nf,3,2024,.01,pools,year,kind,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma,z,qzero)
    brute=[]
    for i in range(n):
        means=[]
        for j in refsets[i]:
            value=0.
            for f in range(nf):
                numerator=denominator=0.
                for other in range(n):
                    if other==i:continue
                    for k in refsets[other]:
                        w=t[other,f];numerator+=w*(1-a[j]@a[k])*(1-t[j]@t[k]);denominator+=w
                value+=t[i,f]*numerator/denominator
            means.append(value)
        brute.append(np.mean(means))
    np.testing.assert_allclose(qzero,brute,atol=1e-7)
    np.testing.assert_allclose(net[0,1].sum(),n,atol=1e-10)
    np.testing.assert_allclose(net[0,2].sum(),z[:,3].sum(),atol=1e-10)
    np.testing.assert_allclose(net[0,5].sum(),np.sum(brute),atol=1e-10)
    assert np.all(net[0,2]<=net[0,1]+1e-12) and nmodel[0]==n
