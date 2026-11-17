from pathlib import Path
import json
import sys
import unittest
import zipfile
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import fbeta_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train import FEATURES,SEED,split_indices,pick_threshold,metrics,portable_predict


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads((ROOT/'dist/data.js').read_text(encoding='utf-8').removeprefix('window.CHURN_DATA=').strip().removesuffix(';'))
        with zipfile.ZipFile(ROOT/'data/iranian-churn.zip') as z:
            with z.open('Customer Churn.csv') as f:cls.raw=pd.read_csv(f)
        cls.raw.columns=[' '.join(c.split()) for c in cls.raw.columns]
        cls.membership=pd.read_csv(ROOT/'data/split-membership.csv',dtype={'profile_group':str})

    def test_split_groups_do_not_overlap(self):
        groups={n:set(self.membership.loc[self.membership['split'].eq(n),'profile_group']) for n in ['train','validation','test']}
        self.assertTrue(groups['train'].isdisjoint(groups['validation']))
        self.assertTrue(groups['train'].isdisjoint(groups['test']))
        self.assertTrue(groups['test'].isdisjoint(groups['validation']))
        self.assertEqual(len(self.membership),3150)
        self.assertEqual(self.membership['row'].nunique(),3150)

    def test_split_is_reproducible_from_source(self):
        X=self.raw[FEATURES].astype(float);groups=pd.util.hash_pandas_object(X,index=False)
        splits=split_indices(X,self.raw['Churn'],groups)
        for name,idx in zip(['train','validation','test'],splits):
            actual=set(self.membership.loc[self.membership['split'].eq(name),'row'])
            self.assertEqual(actual,set(idx))
            self.assertEqual(len(idx),self.d['meta']['splits'][name]['rows'])

    def test_conservative_feature_policy(self):
        self.assertEqual(len(FEATURES),9)
        for excluded in ['Churn','Status','Customer Value','Age','Age Group']:
            self.assertNotIn(excluded,self.d['model']['features'])

    def test_threshold_uses_validation_predictions(self):
        val=pd.read_csv(ROOT/'reports/validation-predictions.csv')
        chosen=pick_threshold(val['actual'].to_numpy(),val['score'].to_numpy())
        self.assertAlmostEqual(chosen,self.d['meta']['threshold'])
        chosen_f2=fbeta_score(val['actual'],val['score']>=chosen,beta=2)
        self.assertTrue(all(chosen_f2>=fbeta_score(val['actual'],val['score']>=t,beta=2,zero_division=0)-1e-12 for t in np.arange(.05,.951,.01)))

    def test_model_portability_and_test_scores(self):
        model=joblib.load(ROOT/'models/churn-model.joblib')
        rows=self.d['testRows'];X=pd.DataFrame([r['x'] for r in rows],columns=FEATURES)
        sklearn_p=model.predict_proba(X)[:,1]
        portable=portable_predict(self.d['model'],X)
        np.testing.assert_allclose(sklearn_p,portable,atol=1e-12)
        np.testing.assert_allclose(portable,[r['score'] for r in rows],atol=1e-12)

    def test_metrics_reconcile(self):
        rows=self.d['testRows'];result=metrics([r['actual'] for r in rows],np.asarray([r['score'] for r in rows]),self.d['meta']['threshold'])
        for key,value in result.items():self.assertAlmostEqual(value,self.d['meta']['test'][key])
        self.assertEqual(result['tp']+result['fn'],self.d['meta']['splits']['test']['churners'])
        self.assertEqual(sum(result[k] for k in ['tp','fp','tn','fn']),len(rows))

    def test_reference_profiles_are_valid_integers(self):
        for f in self.d['fields']:
            self.assertEqual(f['reference'],int(f['reference']))
            self.assertLessEqual(f['min'],f['reference'])
            self.assertGreaterEqual(f['max'],f['reference'])

    def test_group_bootstrap_intervals_are_valid(self):
        for metric,interval in self.d['meta']['intervals'].items():
            self.assertTrue(0<=interval[0]<=self.d['meta']['test'][metric]<=interval[1]<=1)


if __name__=='__main__':unittest.main()
