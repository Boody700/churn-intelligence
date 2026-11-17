"""Reproducible, group-isolated telecom churn modeling and portable inference export."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parents[1]
SEED=42
FEATURES=['Call Failure','Complains','Subscription Length','Charge Amount','Seconds of Use','Frequency of use','Frequency of SMS','Distinct Called Numbers','Tariff Plan']
LABELS=['Call failures','Complaint recorded','Subscription length','Charge band','Voice usage','Call count','SMS count','Distinct called numbers','Tariff plan']
UNITS=['failures / 9 months','0 = no · 1 = yes','months','ordinal charge band','seconds / 9 months','calls / 9 months','messages / 9 months','numbers / 9 months','1 = pay as you go · 2 = contract']


def split_indices(X,y,groups):
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    develop,test=next(outer.split(X,y,groups))
    inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+1)
    train,val=next(inner.split(X.iloc[develop],y.iloc[develop],groups.iloc[develop]))
    return develop[train],develop[val],test


def metrics(y,p,threshold):
    pred=np.asarray(p)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'average_precision':float(average_precision_score(y,p)), 'roc_auc':float(roc_auc_score(y,p)), 'brier':float(brier_score_loss(y,p)), 'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0)), 'f2':float(fbeta_score(y,pred,beta=2,zero_division=0)), 'alerts':int(pred.sum()), 'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)}


def pick_threshold(y,p):
    choices=np.arange(.05,.951,.01)
    return float(max(choices,key=lambda t:(fbeta_score(y,p>=t,beta=2,zero_division=0),precision_score(y,p>=t,zero_division=0),t)))


def export_estimator(estimator):
    if hasattr(estimator,'named_steps'):
        scaler=estimator.named_steps['standardscaler']
        model=estimator.named_steps['logisticregression']
        return {'kind':'logistic','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}
    trees=[]
    for member in estimator.estimators_:
        t=member.tree_
        positive=t.value[:,0,1]/t.value[:,0,:].sum(axis=1)
        trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'positive':positive.tolist()})
    return {'kind':'forest','trees':trees}


def portable_predict(model,X):
    X=np.asarray(X,dtype=float)
    scores=[]
    for fold in model['folds']:
        e=fold['estimator']
        if e['kind']=='logistic':
            score=((X-e['mean'])/e['scale'])@e['coef']+e['intercept']
        else:
            totals=[]
            for row in X:
                tree_probs=[]
                for t in e['trees']:
                    n=0
                    while t['left'][n]!=-1:
                        n=t['left'][n] if np.float32(row[t['feature'][n]])<=t['threshold'][n] else t['right'][n]
                    tree_probs.append(t['positive'][n])
                totals.append(np.mean(tree_probs))
            score=np.asarray(totals)
        scores.append(1/(1+np.exp(np.clip(fold['a']*score+fold['b'],-700,700))))
    return np.mean(scores,axis=0)


def train():
    for name in ['data','models','dist','reports','docs']:(ROOT/name).mkdir(exist_ok=True)
    source=ROOT/'data/iranian-churn.zip'
    if not source.exists():urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip',source)
    with zipfile.ZipFile(source) as z:
        with z.open('Customer Churn.csv') as f:df=pd.read_csv(f)
    df.columns=[' '.join(c.split()) for c in df.columns]
    assert len(df)==3150 and df.isna().sum().sum()==0
    X=df[FEATURES].astype(float);y=df['Churn'].astype(int)
    groups=pd.util.hash_pandas_object(X,index=False)
    train_idx,val_idx,test_idx=split_indices(X,y,groups)
    for a,b in [(train_idx,val_idx),(train_idx,test_idx),(val_idx,test_idx)]:assert set(groups.iloc[a]).isdisjoint(groups.iloc[b])
    cv=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+2).split(X.iloc[train_idx],y.iloc[train_idx],groups.iloc[train_idx]))
    candidates={
        'Logistic / C=0.1':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=SEED)),
        'Logistic / C=1':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=SEED)),
        'Random forest / depth 6':RandomForestClassifier(n_estimators=120,max_depth=6,min_samples_leaf=6,random_state=SEED,n_jobs=1),
        'Random forest / depth 10':RandomForestClassifier(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=SEED,n_jobs=1),
    }
    comparison=[]
    for name,model in candidates.items():
        scores=cross_val_score(model,X.iloc[train_idx],y.iloc[train_idx],cv=cv,scoring='average_precision',n_jobs=1)
        comparison.append({'name':name,'cv_ap':float(scores.mean()),'cv_std':float(scores.std())})
        print(name,round(scores.mean(),4),flush=True)
    selected=max(comparison,key=lambda r:r['cv_ap'])['name']
    calibrated=CalibratedClassifierCV(candidates[selected],method='sigmoid',cv=cv,ensemble=True)
    calibrated.fit(X.iloc[train_idx],y.iloc[train_idx])
    val_p=calibrated.predict_proba(X.iloc[val_idx])[:,1]
    threshold=pick_threshold(y.iloc[val_idx],val_p)
    # The held-out test set is scored only after model and threshold selection.
    test_p=calibrated.predict_proba(X.iloc[test_idx])[:,1]
    portable={'features':FEATURES,'folds':[{'estimator':export_estimator(c.estimator),'a':float(c.calibrators[0].a_),'b':float(c.calibrators[0].b_)} for c in calibrated.calibrated_classifiers_]}
    reconstructed=portable_predict(portable,X.iloc[test_idx])
    max_error=float(np.max(np.abs(test_p-reconstructed)))
    assert max_error<1e-10,max_error
    joblib.dump(calibrated,ROOT/'models/churn-model.joblib')
    (ROOT/'models/portable-model.json').write_text(json.dumps(portable,separators=(',',':')),encoding='utf-8')
    importance=permutation_importance(calibrated,X.iloc[val_idx],y.iloc[val_idx],n_repeats=8,random_state=SEED,scoring='average_precision')
    rng=np.random.default_rng(SEED)
    test_groups=groups.iloc[test_idx].to_numpy()
    unique=np.unique(test_groups)
    group_members={g:np.flatnonzero(test_groups==g) for g in unique}
    bootstrap=[]
    yt=y.iloc[test_idx].to_numpy()
    for _ in range(400):
        indices=np.concatenate([group_members[g] for g in rng.choice(unique,size=len(unique),replace=True)])
        if len(np.unique(yt[indices]))==2:bootstrap.append([average_precision_score(yt[indices],test_p[indices]),roc_auc_score(yt[indices],test_p[indices])])
    interval=np.quantile(bootstrap,[.025,.975],axis=0)
    splits={name:{'rows':len(idx),'groups':int(groups.iloc[idx].nunique()),'churners':int(y.iloc[idx].sum()),'prevalence':float(y.iloc[idx].mean())} for name,idx in [('train',train_idx),('validation',val_idx),('test',test_idx)]}
    fields=[{'name':f,'label':label,'unit':unit,'min':float(X.iloc[train_idx][f].min()),'max':float(X.iloc[train_idx][f].max()),'reference':int(np.rint(X.iloc[train_idx][f].median())),'importance':float(importance.importances_mean[i]),'importance_std':float(importance.importances_std[i])} for i,(f,label,unit) in enumerate(zip(FEATURES,LABELS,UNITS))]
    test_rows=[{'id':f'TEST-{i+1:04d}','x':X.iloc[idx].tolist(),'actual':int(y.iloc[idx]),'score':float(p),'group':str(test_groups[i])} for i,(idx,p) in enumerate(zip(test_idx,test_p))]
    # Stable order supports a ranked queue and deterministic ties.
    test_rows.sort(key=lambda r:(-r['score'],r['id']))
    precision,recall,pr_t=precision_recall_curve(yt,test_p)
    fpr,tpr,_=roc_curve(yt,test_p)
    mean_pred,frac_actual=None,None
    frac_actual,mean_pred=calibration_curve(yt,test_p,n_bins=8,strategy='quantile')
    meta={'source_rows':len(df),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'unique_profiles':int(groups.nunique()),'exact_duplicate_rows':int(df.duplicated().sum()),'selected_model':selected,'selection':'Highest mean grouped 3-fold training average precision','seed':SEED,'splits':splits,'threshold':threshold,'validation':metrics(y.iloc[val_idx],val_p,threshold),'test':metrics(yt,test_p,threshold),'baseline':metrics(yt,np.full(len(yt),y.iloc[train_idx].mean()),threshold),'intervals':{'average_precision':interval[:,0].tolist(),'roc_auc':interval[:,1].tolist()},'browser_parity_max_error':max_error,'excluded_features':['Status','Customer Value','Age','Age Group'],'versions':{'sklearn':__import__('sklearn').__version__,'pandas':pd.__version__,'numpy':np.__version__}}
    payload={'meta':meta,'fields':fields,'comparison':comparison,'model':portable,'testRows':test_rows,'curves':{'pr':list(zip(recall.tolist(),precision.tolist())),'roc':list(zip(fpr.tolist(),tpr.tolist())),'calibration':list(zip(mean_pred.tolist(),frac_actual.tolist()))},'parity':[{'x':X.iloc[i].tolist(),'expected':float(p)} for i,p in zip(test_idx[:30],test_p[:30])]}
    (ROOT/'dist/data.js').write_text('window.CHURN_DATA='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';\n',encoding='utf-8')
    (ROOT/'data/metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    membership=pd.DataFrame({'row':np.arange(len(df)),'split':'','profile_group':groups.astype(str)})
    for name,idx in [('train',train_idx),('validation',val_idx),('test',test_idx)]:membership.loc[idx,'split']=name
    membership.to_csv(ROOT/'data/split-membership.csv',index=False)
    pd.DataFrame(comparison).to_csv(ROOT/'reports/model-comparison.csv',index=False)
    pd.DataFrame(fields).to_csv(ROOT/'reports/permutation-importance.csv',index=False)
    test_export=pd.DataFrame(test_rows).drop(columns=['x','group'])
    test_export.to_csv(ROOT/'reports/test-predictions.csv',index=False)
    pd.DataFrame({'actual':y.iloc[val_idx].to_numpy(),'score':val_p}).to_csv(ROOT/'reports/validation-predictions.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'churn-intelligence-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    fig.set_facecolor('#f7f8fc')
    axes[0].plot(recall,precision,color='#6659c4',lw=2);axes[0].axhline(yt.mean(),color='#939bad',ls='--',label='Test prevalence');axes[0].set(xlabel='Recall',ylabel='Precision',title=f"Precision–recall · AP {meta['test']['average_precision']:.3f}",xlim=(0,1),ylim=(0,1));axes[0].legend(fontsize=8)
    axes[1].plot(fpr,tpr,color='#6659c4',lw=2);axes[1].plot([0,1],[0,1],'--',color='#939bad');axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title=f"ROC · AUC {meta['test']['roc_auc']:.3f}",xlim=(0,1),ylim=(0,1))
    axes[2].plot(mean_pred,frac_actual,'o-',color='#6659c4');axes[2].plot([0,1],[0,1],'--',color='#939bad');axes[2].set(xlabel='Mean predicted risk',ylabel='Observed churn fraction',title=f"Calibration · Brier {meta['test']['brier']:.3f}",xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.supxlabel('Untouched grouped test partition · UCI Iranian Churn / CC BY 4.0 · No claim of production or causal performance',fontsize=8)
    fig.savefig(ROOT/'reports/evaluation.png',dpi=170);plt.close(fig)
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':train()
