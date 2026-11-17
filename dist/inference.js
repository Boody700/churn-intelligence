(function(root){
  'use strict';
  function predict(model,x){
    if(x.length!==model.features.length||!x.every(Number.isFinite))throw new Error('Provide one finite number for each model feature.');
    let result=0;
    for(const fold of model.folds){
      const e=fold.estimator;let score;
      if(e.kind==='logistic')score=e.intercept+x.reduce((a,v,i)=>a+(v-e.mean[i])/e.scale[i]*e.coef[i],0);
      else if(e.kind==='forest'){
        score=0;
        for(const tree of e.trees){let node=0;while(tree.left[node]!==-1)node=Math.fround(x[tree.feature[node]])<=tree.threshold[node]?tree.left[node]:tree.right[node];score+=tree.positive[node];}
        score/=e.trees.length;
      }else throw new Error('Unknown estimator type');
      result+=1/(1+Math.exp(Math.max(-700,Math.min(700,fold.a*score+fold.b))));
    }
    return result/model.folds.length;
  }
  function metrics(rows,threshold){
    let tp=0,fp=0,tn=0,fn=0;for(const r of rows){if(r.score>=threshold){if(r.actual)tp++;else fp++;}else if(r.actual)fn++;else tn++;}
    const precision=tp+fp?tp/(tp+fp):0,recall=tp+fn?tp/(tp+fn):0;
    return {tp,fp,tn,fn,precision,recall,alerts:tp+fp,f2:4*precision+recall?5*precision*recall/(4*precision+recall):0};
  }
  function sensitivity(model,x,fields){const base=predict(model,x);return fields.map((f,i)=>{const changed=x.slice();changed[i]=f.reference;return {index:i,label:f.label,reference:f.reference,delta:base-predict(model,changed)};}).sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));}
  const api={predict,metrics,sensitivity};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ChurnEngine=api;
})(typeof window!=='undefined'?window:globalThis);
