const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const E=require('../dist/inference.js');
const nodes=new Map();let saved;
class Element{constructor(){this.value='';this.innerHTML='';this.textContent='';this.events={};this.dataset={};this.classList={toggle(){},add(){},remove(){}};this.attrs={};}addEventListener(e,fn){this.events[e]=fn;}setAttribute(k,v){this.attrs[k]=v;}removeAttribute(k){delete this.attrs[k];}focus(){}click(){}remove(){}append(){}}
const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const links=['overview','workbench','threshold','model-card'].map(page=>Object.assign(new Element(),{dataset:{page}}));
const window={ChurnEngine:E,events:{},addEventListener(k,f){this.events[k]=f;},scrollTo(){}};
const location={hash:'#workbench'};
const context={window,location,document:{querySelector:s=>get(s.slice(1)),querySelectorAll:()=>links,getElementById:get,createElement:()=>new Element(),body:new Element()},Blob,Intl,URL:{createObjectURL(b){saved=b;return 'blob:test';},revokeObjectURL(){}},setTimeout(fn){fn();},console};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist/data.js'),'utf8'),context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist/app.js'),'utf8'),context);
const D=window.CHURN_DATA,route=page=>{location.hash='#'+page;window.events.hashchange();};
(async()=>{
  assert.ok(get('content').innerHTML.includes('No input has a score difference'));
  D.fields.forEach((f,i)=>get('feature-'+i).value=String(f.reference));
  get('feature-0').value='12';get('feature-0').events.input();
  get('export-profile').events.click();let report=JSON.parse(await saved.text());assert.equal(report.inputs['Call Failure'],D.fields[0].reference,'Export must describe last calculation, not unsaved inputs');
  get('risk-form').events.submit({preventDefault(){}});get('export-profile').events.click();report=JSON.parse(await saved.text());assert.equal(report.inputs['Call Failure'],12);
  get('feature-4').value='';get('feature-4').events.input();route('overview');route('workbench');assert.ok(get('content').innerHTML.includes('id="feature-4" value=""'),'Blank numeric input must remain blank across navigation');
  get('risk-form').events.submit({preventDefault(){}});assert.ok(get('form-error').textContent.includes('whole numbers'));
  route('overview');assert.ok(get('content').innerHTML.includes('Read the chart data'));
  route('model-card');get('export-model-card').events.click();assert.equal(JSON.parse(await saved.text()).source_sha256,D.meta.source_sha256);
  route('threshold');get('threshold').value='50';get('threshold').events.input({target:get('threshold')});assert.equal(get('threshold').attrs['aria-valuetext'],'50% estimated-risk cutoff');
  console.log('PASS: blank-input preservation, last-calculation exports, reference empty state, chart alternatives, metadata export and threshold accessibility.');
})().catch(error=>{console.error(error);process.exitCode=1;});
