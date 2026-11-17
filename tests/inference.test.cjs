const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const E=require('../dist/inference.js');
const sandbox={window:{}};vm.createContext(sandbox);vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist/data.js'),'utf8'),sandbox);
const D=JSON.parse(JSON.stringify(sandbox.window.CHURN_DATA));
let maxError=0;
for(const r of D.testRows){const p=E.predict(D.model,r.x);maxError=Math.max(maxError,Math.abs(p-r.score));assert.ok(p>=0&&p<=1);}
assert.ok(maxError<1e-10,`Prediction mismatch: ${maxError}`);
const m=E.metrics(D.testRows,D.meta.threshold);
for(const key of ['tp','fp','tn','fn','alerts','precision','recall','f2'])assert.ok(Math.abs(m[key]-D.meta.test[key])<1e-12,`Metric mismatch: ${key}`);
assert.equal(m.tp+m.fp+m.tn+m.fn,D.testRows.length);
assert.equal(E.metrics(D.testRows,0).alerts,D.testRows.length);
assert.equal(E.metrics(D.testRows,1).alerts,0);
let previous=1;
for(let threshold=0;threshold<=1;threshold+=.01){const result=E.metrics(D.testRows,threshold);assert.ok(result.recall<=previous);previous=result.recall;}
assert.throws(()=>E.predict(D.model,[NaN]));
assert.throws(()=>E.predict(D.model,new Array(D.fields.length).fill(Infinity)));
const reference=D.fields.map(f=>f.reference);
assert.ok(reference.every(Number.isInteger));
assert.ok(E.sensitivity(D.model,reference,D.fields).every(r=>r.delta===0));
console.log(`PASS: all ${D.testRows.length} held-out predictions match Python; maximum error ${maxError}. Threshold metrics, conservation, monotonic recall, reference sensitivities and invalid inputs checked.`);
