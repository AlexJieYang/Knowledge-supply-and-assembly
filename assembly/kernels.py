"""Compiled loops only where ragged reference sets make pandas expansion costly.
No fastmath, nested thread pools or explicit reference-pair materialisation.
"""
import numpy as np
from numba import njit,prange

@njit(cache=True)
def increment(counts,rows):
    for i in rows: counts[i]+=1

@njit(cache=True)
def scatter(cursor,target,rows,values):
    for k in range(len(rows)):
        i=rows[k];target[cursor[i]]=values[k];cursor[i]+=1

@njit(cache=True,parallel=True)
def sort_unique_rows(ptr,index,counts):
    for i in prange(len(counts)):
        a=ptr[i];b=ptr[i+1]
        if a==b:counts[i]=0;continue
        row=index[a:b];row.sort();n=1
        for k in range(1,len(row)):
            if row[k]!=row[n-1]:row[n]=row[k];n+=1
        counts[i]=n

@njit(cache=True)
def membership_sum(ptr,code,weight,j,tot):
    for k in range(ptr[j],ptr[j+1]):tot[code[k]]+=weight[k]

@njit(cache=True)
def age_stratum(age,kind):
    if age==0:a=0
    elif age<=2:a=1
    elif age<=5:a=2
    elif age<=10:a=3
    elif age<=20:a=4
    elif age<=40:a=5
    else:a=6
    return a*3+kind

@njit(cache=True,parallel=True)
def metric_batch(start,stop,blocks,nc,nf,maxrefs,year,primary,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma):
    out=np.full((stop-start,8),np.nan)
    step=(stop-start+blocks-1)//blocks
    for block in prange(blocks):
        lo=start+block*step;hi=min(stop,lo+step)
        A=np.zeros(nc);T=np.zeros(nf);V=np.zeros(nc*nf)
        PT=np.zeros(nf);PV=np.zeros(nc*nf)
        positions=np.empty(maxrefs,np.int64)
        for i in range(lo,hi):
            m=0;allprimary=True;ages=0.
            for pos in range(rp[i],rp[i]+rn[i]):
                j=ri[pos]
                if cp[j]==cp[j+1] or fp[j]==fp[j+1]:continue
                positions[m]=pos;m+=1;ages+=year[i]-year[j]
                membership_sum(cp,cc,cw,j,A);membership_sum(fp,fc,fw,j,T)
                pj=primary[j]
                if pj<0:allprimary=False
                else:PT[pj]+=1.
                for u in range(cp[j],cp[j+1]):
                    c=cc[u];wc=cw[u]
                    if pj>=0:PV[c*nf+pj]+=wc
                    for v in range(fp[j],fp[j+1]):V[c*nf+fc[v]]+=wc*fw[v]
            o=i-start;out[o,0]=m
            if m>=2:
                gs=ds=qs=ps=0.
                for k in range(m):
                    pos=positions[k];j=ri[pos];ad=td=vd=0.;ap=0.;pj=primary[j]
                    for u in range(cp[j],cp[j+1]):
                        c=cc[u];wc=cw[u];ad+=wc*(A[c]-wc)
                        if pj>=0:ap+=wc*(PV[c*nf+pj]-wc)
                        for v in range(fp[j],fp[j+1]):
                            f=fc[v];wf=fw[v];vd+=wc*wf*(V[c*nf+f]-wc*wf)
                    for v in range(fp[j],fp[j+1]):
                        f=fc[v];wf=fw[v];td+=wf*(T[f]-wf)
                    g=min(1.,max(0.,1-ad/(m-1)));d=min(1.,max(0.,1-td/(m-1)))
                    s=1-(ad+td-vd)/(m-1)
                    s=min(1.,max(0.,s));sigma[pos]=s
                    gs+=g;ds+=d;qs+=float(sigma[pos])
                    if allprimary:ps+=1-(ad+(PT[pj]-1)-ap)/(m-1)
                q=qs/m;out[o,1]=gs/m;out[o,2]=ds/m;out[o,3]=q
                if allprimary:out[o,4]=min(1.,max(0.,ps/m))
                out[o,5]=ages/m
                total=float(rn[i])*(rn[i]-1)/2;observed=float(m)*(m-1)/2*q
                if total>0:
                    out[o,6]=observed/total
                    out[o,7]=(observed+total-float(m)*(m-1)/2)/total
            # Clear only touched coordinates, not nc*nf cells for every paper.
            for k in range(m):
                j=ri[positions[k]];pj=primary[j]
                if pj>=0:PT[pj]=0.
                for u in range(cp[j],cp[j+1]):
                    c=cc[u];A[c]=0.
                    if pj>=0:PV[c*nf+pj]=0.
                    for v in range(fp[j],fp[j+1]):V[c*nf+fc[v]]=0.
                for v in range(fp[j],fp[j+1]):T[fc[v]]=0.
    return out

@njit(cache=True,parallel=True)
def build_pools(group_ptr,group_rows,nc,nf,year,kind,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma):
    ny=len(group_ptr)-1;dim=2+nc+nf+nc*nf
    pools=np.zeros((ny,nf,21,dim))
    for y in prange(ny):
        for z in range(group_ptr[y],group_ptr[y+1]):
            i=group_rows[z]
            for pos in range(rp[i],rp[i]+rn[i]):
                j=ri[pos]
                if cp[j]==cp[j+1] or fp[j]==fp[j+1]:continue
                h=age_stratum(year[i]-year[j],kind[j])
                for fpos in range(fp[i],fp[i+1]):
                    f=fc[fpos];w=fw[fpos];s=pools[y,f,h]
                    s[0]+=w;s[-1]+=w*sigma[pos]
                    for u in range(cp[j],cp[j+1]):
                        c=cc[u];wc=cw[u];s[1+c]+=w*wc
                        for v in range(fp[j],fp[j+1]):s[1+nc+nf+c*nf+fc[v]]+=w*wc*fw[v]
                    for v in range(fp[j],fp[j+1]):s[1+nc+fc[v]]+=w*fw[v]
    return pools

@njit(cache=True,parallel=True)
def aggregate_networks(group_ptr,group_rows,nc,nf,maxrefs,endyear,minpool,pools,
                       year,kind,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma,metrics,model_q):
    ny=len(group_ptr)-1
    # C is a symmetric adjacency: sum(C)/2 equals the international paper count.
    net=np.zeros((ny,7,nc,nc));role=np.zeros((ny,12,nc,nf))
    cohort=np.zeros((ny,endyear+1,2));nmodel=np.zeros(ny,np.int64)
    for y in prange(ny):
        LA=np.zeros((21,nc));LT=np.zeros((21,nf));LV=np.zeros((21,nc*nf));L=np.zeros(21)
        positions=np.empty(maxrefs,np.int64);hs=np.empty(maxrefs,np.int64)
        s0=np.empty(maxrefs);sc=np.empty(maxrefs)
        for z in range(group_ptr[y],group_ptr[y+1]):
            i=group_rows[z];m=0
            for pos in range(rp[i],rp[i]+rn[i]):
                j=ri[pos]
                if cp[j]==cp[j+1] or fp[j]==fp[j+1]:continue
                h=age_stratum(year[i]-year[j],kind[j]);positions[m]=pos;hs[m]=h;m+=1;L[h]+=1.
                for u in range(cp[j],cp[j+1]):
                    c=cc[u];wc=cw[u];LA[h,c]+=wc
                    for v in range(fp[j],fp[j+1]):LV[h,c*nf+fc[v]]+=wc*fw[v]
                for v in range(fp[j],fp[j+1]):LT[h,fc[v]]+=fw[v]
            valid=True
            for k in range(m):
                j=ri[positions[k]];hj=hs[k];expect=0.;context=0.
                for fpos in range(fp[i],fp[i+1]):
                    f=fc[fpos];wfocal=fw[fpos];pool=pools[y,f,hj]
                    context+=wfocal*pool[-1]/pool[0]
                    for h in range(21):
                        partners=L[h]-(1 if h==hj else 0)
                        if partners<=0:continue
                        p=pools[y,f,h];den=p[0]-wfocal*L[h]
                        if den<minpool:valid=False;continue
                        e=1.
                        for u in range(cp[j],cp[j+1]):
                            c=cc[u];wc=cw[u];e-=wc*(p[1+c]-wfocal*LA[h,c])/den
                            for v in range(fp[j],fp[j+1]):
                                fref=fc[v];wref=fw[v]
                                e+=wc*wref*(p[1+nc+nf+c*nf+fref]-wfocal*LV[h,c*nf+fref])/den
                        for v in range(fp[j],fp[j+1]):
                            fref=fc[v];wref=fw[v];e-=wref*(p[1+nc+fref]-wfocal*LT[h,fref])/den
                        expect+=wfocal*partners/(m-1)*e
                s0[k]=min(1.,max(0.,expect));sc[k]=context
            q=metrics[i,3];qzero=np.sum(s0[:m])/m;qcontext=np.sum(sc[:m])/m
            if valid:model_q[i]=qzero;nmodel[y]+=1
            kc=cp[i+1]-cp[i]
            if kc>=2:
                wc=2./(kc*(kc-1))
                for u in range(cp[i],cp[i+1]):
                    for v in range(u+1,cp[i+1]):
                        a=cc[u];b=cc[v];net[y,0,a,b]+=wc;net[y,0,b,a]+=wc
            for u in range(cp[i],cp[i+1]):
                b=cc[u];wb=cw[u]
                for v in range(fp[i],fp[i+1]):
                    f=fc[v];w=wb*fw[v];role[y,2,b,f]+=w;role[y,3,b,f]+=w*q
                    role[y,11,b,f]+=w*qcontext
                    if valid:
                        role[y,7,b,f]+=w;role[y,8,b,f]+=w*q;role[y,9,b,f]+=w*qzero
            for k in range(m):
                j=ri[positions[k]];s=sigma[positions[k]];e=1./m;bcredit=s/m
                cohort[y,year[j],0]+=e;cohort[y,year[j],1]+=bcredit
                for u in range(cp[j],cp[j+1]):
                    a=cc[u];wa=cw[u]
                    for v in range(fp[j],fp[j+1]):
                        f=fc[v];w=wa*fw[v]/m
                        role[y,0,a,f]+=w;role[y,1,a,f]+=w*s;role[y,10,a,f]+=w*sc[k]
                        if valid:
                            role[y,4,a,f]+=w;role[y,5,a,f]+=w*s;role[y,6,a,f]+=w*s0[k]
                    for v in range(cp[i],cp[i+1]):
                        b=cc[v];w=wa*cw[v]/m
                        net[y,1,a,b]+=w;net[y,2,a,b]+=w*s;net[y,6,a,b]+=w*sc[k]
                        if valid:
                            net[y,3,a,b]+=w;net[y,4,a,b]+=w*s;net[y,5,a,b]+=w*s0[k]
            for k in range(m):
                j=ri[positions[k]];h=hs[k];L[h]=0.
                for u in range(cp[j],cp[j+1]):
                    c=cc[u];LA[h,c]=0.
                    for v in range(fp[j],fp[j+1]):LV[h,c*nf+fc[v]]=0.
                for v in range(fp[j],fp[j+1]):LT[h,fc[v]]=0.
    return net,role,cohort,nmodel

@njit(cache=True,parallel=True)
def field_summaries(gp,gr,nf,fp,fc,fw,metrics):
    # Per metric: fractional count, sum, sum of squares.
    out=np.zeros((len(gp)-1,nf,6,3))
    for y in prange(len(gp)-1):
        for p in range(gp[y],gp[y+1]):
            i=gr[p]
            for k in range(fp[i],fp[i+1]):
                f=fc[k];w=fw[k]
                for mi,met in enumerate((1,2,3,4,5,-1)):
                    x=metrics[i,met] if met>=0 else metrics[i,3]-metrics[i,1]*metrics[i,2]
                    if np.isfinite(x):out[y,f,mi,0]+=w;out[y,f,mi,1]+=w*x;out[y,f,mi,2]+=w*x*x
    return out

@njit(cache=True,parallel=True)
def core_output(gp,gr,nc,nf,cp,cc,cw,fp,fc,fw):
    out=np.zeros((len(gp)-1,nc,nf))
    for y in prange(len(gp)-1):
        for z in range(gp[y],gp[y+1]):
            i=gr[z]
            for u in range(cp[i],cp[i+1]):
                for v in range(fp[i],fp[i+1]):out[y,cc[u],fc[v]]+=cw[u]*fw[v]
    return out
