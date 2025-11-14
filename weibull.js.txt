/* weibull.js
   Client-side Weibull MLE (censored), bootstrap CI, Plotly plots, PM cost
   Converted from Python notebook.
*/

// ---------- Helpers ----------
function parseTTF(text){
  const failures=[], susp=[];
  const lines = text.split('\n').map(s=>s.trim()).filter(Boolean);
  for(const ln of lines){
    const parts = ln.split(',');
    if(parts.length<2) continue;
    const t = parseFloat(parts[0]);
    const code = parts[1].trim().toUpperCase();
    if(isNaN(t)) continue;
    if(code==='F') failures.push(t);
    else if(code==='S') susp.push(t);
  }
  return {failures, susp};
}

function R_weibull(t, beta, eta){ return Math.exp(-Math.pow(t/eta, beta)); }
function pdf_weibull(t, beta, eta){ return (beta/eta) * Math.pow(t/eta, beta-1) * Math.exp(-Math.pow(t/eta, beta)); }
function hazard_weibull(t,beta,eta){ return (beta/eta) * Math.pow(t/eta, beta-1); }

// ---------- Nelder-Mead (compact) ----------
function nelderMead(f, x0, options){
  const maxIter = options.maxIter||2000;
  const tol = options.tol||1e-6;
  const alpha=1, gamma=2, rho=0.5, sigma=0.5;
  let simplex=[x0, [x0[0]+0.1*Math.max(1,Math.abs(x0[0])), x0[1]], [x0[0], x0[1]+0.1*Math.max(1,Math.abs(x0[1]))]];
  let fx = simplex.map(p=>f(p));
  for(let iter=0; iter<maxIter; iter++){
    const idx = fx.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]).map(x=>x[1]);
    simplex = [simplex[idx[0]], simplex[idx[1]], simplex[idx[2]]];
    fx = [fx[idx[0]], fx[idx[1]], fx[idx[2]]];
    const x0m = [(simplex[0][0]+simplex[1][0])/2, (simplex[0][1]+simplex[1][1])/2];
    const xr = [x0m[0] + alpha*(x0m[0]-simplex[2][0]), x0m[1] + alpha*(x0m[1]-simplex[2][1])];
    const fxr = f(xr);
    if(fxr < fx[0]){
      const xe = [x0m[0] + gamma*(xr[0]-x0m[0]), x0m[1] + gamma*(xr[1]-x0m[1])];
      const fxe = f(xe);
      if(fxe < fxr){ simplex[2]=xe; fx[2]=fxe; }
      else { simplex[2]=xr; fx[2]=fxr; }
    } else if(fxr < fx[1]){
      simplex[2]=xr; fx[2]=fxr;
    } else {
      const xc = [x0m[0] + rho*(simplex[2][0]-x0m[0]), x0m[1] + rho*(simplex[2][1]-x0m[1])];
      const fxc = f(xc);
      if(fxc < fx[2]){
        simplex[2]=xc; fx[2]=fxc;
      } else {
        simplex[1] = [simplex[0][0] + sigma*(simplex[1][0]-simplex[0][0]), simplex[0][1] + sigma*(simplex[1][1]-simplex[0][1])];
        simplex[2] = [simplex[0][0] + sigma*(simplex[2][0]-simplex[0][0]), simplex[0][1] + sigma*(simplex[2][1]-simplex[0][1])];
        fx = [f(simplex[0]), f(simplex[1]), f(simplex[2])];
      }
    }
    const fmean = (fx[0]+fx[1]+fx[2])/3;
    const ss = Math.sqrt(((fx[0]-fmean)**2 + (fx[1]-fmean)**2 + (fx[2]-fmean)**2)/3);
    if(ss < tol) break;
  }
  const bestIdx = fx[0] <= fx[1] && fx[0] <= fx[2] ? 0 : (fx[1] <= fx[2] ? 1 : 2);
  return {x: simplex[bestIdx], fx: fx[bestIdx]};
}

// ---------- Likelihood (censored) ----------
function negLogLik(params, failures, susp){
  let beta = params[0], eta = params[1];
  if(beta <= 1e-6 || eta <= 1e-6) return 1e12;
  let ll = 0;
  if(failures.length>0){
    for(const t of failures){
      ll += Math.log(beta/eta) + (beta-1)*Math.log(t/eta) - Math.pow(t/eta, beta);
    }
  }
  if(susp.length>0){
    for(const s of susp){
      ll += - Math.pow(s/eta, beta);
    }
  }
  return -ll;
}

// ---------- Fit Weibull ----------
function fitWeibull(failures, susp){
  const meanF = failures.length>0 ? failures.reduce((a,b)=>a+b,0)/failures.length : 1000;
  const x0 = [1.5, meanF || 1000];
  const res = nelderMead(p => negLogLik(p, failures, susp), x0, {maxIter:1500, tol:1e-8});
  return {beta: Math.max(1e-6, res.x[0]), eta: Math.max(1e-6, res.x[1]), fx:res.fx};
}

// ---------- Bootstrap CI ----------
async function bootstrapCI(failures, susp, nboot, conf=0.90){
  const events = [];
  for(const t of failures) events.push({t, censored:false});
  for(const t of susp) events.push({t, censored:true});
  if(events.length<3) return null;
  const boots = [];
  for(let i=0;i<nboot;i++){
    const sample=[];
    for(let j=0;j<events.length;j++){
      sample.push(events[Math.floor(Math.random()*events.length)]);
    }
    const sf = sample.filter(x=>!x.censored).map(x=>x.t);
    const ss = sample.filter(x=>x.censored).map(x=>x.t);
    const res = fitWeibull(sf, ss);
    if(isFinite(res.beta) && isFinite(res.eta)) boots.push([res.beta, res.eta]);
    if(i%50===0) await new Promise(r=>setTimeout(r,0));
  }
  const betas = boots.map(b=>b[0]).sort((a,b)=>a-b);
  const etas  = boots.map(b=>b[1]).sort((a,b)=>a-b);
  const low = Math.floor((1-conf)/2 * boots.length);
  const high = Math.ceil((1 - (1-conf)/2) * boots.length);
  return {
    beta_ci: [betas[Math.max(0,low)], betas[Math.min(boots.length-1,high-1)]],
    eta_ci:  [etas[Math.max(0,low)], etas[Math.min(boots.length-1,high-1)]],
    raw: boots
  };
}

// ---------- Numeric integration (trapezoid) ----------
function integrateR(Rfunc, args, a, b, n=400){
  const h = (b-a)/n;
  let s=0;
  for(let i=0;i<=n;i++){
    const t = a + i*h;
    s += Rfunc(t, ...args);
  }
  return s * h;
}
function CPM_num(t, beta, eta, C_PM){
  if(t<=0) return Infinity;
  const integral = integrateR(R_weibull, [beta,eta], 0, t, 300);
  return C_PM * R_weibull(t,beta,eta)/integral;
}
function CCM_num(t, beta, eta, C_CM){
  if(t<=0) return Infinity;
  const integral = integrateR(R_weibull, [beta,eta], 0, t, 300);
  const F = 1 - R_weibull(t,beta,eta);
  return C_CM * F / integral;
}
function CPUT_num(t, beta, eta, C_PM, C_CM){
  if(t<=0) return Infinity;
  const integral = integrateR(R_weibull, [beta,eta], 0, t, 300);
  const F = 1 - R_weibull(t,beta,eta);
  return (C_PM * R_weibull(t,beta,eta) + C_CM * F) / integral;
}

// ---------- UI wiring & plotting ----------
document.addEventListener('DOMContentLoaded', ()=>{

  const dataMode = document.getElementById('dataMode');
  const weibullBox = document.getElementById('weibullParams');
  const manualBox = document.getElementById('manualBox');
  const betaEl = document.getElementById('beta');
  const etaEl = document.getElementById('eta');
  const ttfEl = document.getElementById('ttf');
  const fitBtn = document.getElementById('fitBtn');
  const plotBtn = document.getElementById('plotBtn');
  const pmBtn = document.getElementById('pmBtn');
  const exportCsvBtn = document.getElementById('exportCsv');
  const clearBtn = document.getElementById('clearBtn');
  const cpmEl = document.getElementById('cpm');
  const ccmEl = document.getElementById('ccm');
  const tquery = document.getElementById('tquery');
  const confEl = document.getElementById('conf');
  const nbootEl = document.getElementById('nboot');
  const summary = document.getElementById('summary');

  const plotRel = document.getElementById('plotReliability');
  const plotHaz = document.getElementById('plotFailureRate');
  const plotCost = document.getElementById('plotCost');

  function updateMode(){
    if(dataMode.value==='manual'){ manualBox.style.display='block'; weibullBox.style.display='none'; }
    else { manualBox.style.display='none'; weibullBox.style.display='block'; }
  }
  dataMode.addEventListener('change', updateMode);
  updateMode();

  // Fit from manual data
  fitBtn.addEventListener('click', async ()=>{
    const data = parseTTF(ttfEl.value);
    if(data.failures.length===0 && data.susp.length===0){ alert('No data'); return; }
    summary.textContent = 'Estimating...';
    await new Promise(r=>setTimeout(r,10));
    const res = fitWeibull(data.failures, data.susp);
    betaEl.value = res.beta.toFixed(4);
    etaEl.value = res.eta.toFixed(4);
    summary.textContent = `Estimated: β=${res.beta.toFixed(4)}, η=${res.eta.toFixed(4)}`;
  });

  // Plot reliability & hazard
  plotBtn.addEventListener('click', async ()=>{
    try{
      let beta = parseFloat(betaEl.value), eta = parseFloat(etaEl.value);
      const conf = parseFloat(confEl.value), nboot = parseInt(nbootEl.value);
      const tQ = parseFloat(tquery.value);

      if(dataMode.value==='manual'){
        const data = parseTTF(ttfEl.value);
        if(data.failures.length>0 || data.susp.length>0){
          const r = fitWeibull(data.failures, data.susp);
          beta = r.beta; eta = r.eta;
          betaEl.value = beta; etaEl.value = eta;
        }
      }

      summary.textContent = 'Computing curves...';
      const tmax = Math.max(eta*1.6, tQ*1.2, 10);
      const t = Array.from({length:400},(_,i)=> i*(tmax/399));
      const R = t.map(tt=>R_weibull(tt,beta,eta));
      const h = t.map(tt=>hazard_weibull(tt,beta,eta));

      let ci=null;
      if(dataMode.value==='manual'){
        const data = parseTTF(ttfEl.value);
        if(data.failures.length + data.susp.length >= 4){
          summary.textContent = 'Bootstrap CI (may take a few seconds)...';
          ci = await bootstrapCI(data.failures, data.susp, nboot, conf);
        }
      }

      const traces = [{x:t, y:R, name:'R(t)', mode:'lines', line:{color:'black'}}];
      if(ci){
        const [bLow,bHigh] = ci.beta_ci;
        const [eLow,eHigh] = ci.eta_ci;
        const Rlow = t.map(tt=>R_weibull(tt,bHigh,eHigh));
        const Rupp = t.map(tt=>R_weibull(tt,bLow,eLow));
        traces.push({x:[...t,...t.slice().reverse()], y:[...Rupp,...Rlow.slice().reverse()], fill:'toself', fillcolor:'rgba(0,0,0,0.09)', line:{width:0}, name:'CI'});
      }
      Plotly.newPlot(plotRel, traces, {margin:{t:30}, title:`Reliability R(t) — β=${beta.toFixed(3)}, η=${eta.toFixed(3)}`});
      Plotly.newPlot(plotHaz, [{x:t,y:h,mode:'lines',name:'λ(t)'}], {margin:{t:30}, title:'Failure rate λ(t)'});
      summary.textContent = 'Done.';
    }catch(err){
      summary.textContent = 'Error: '+err.message;
    }
  });

  // PM optimization & cost plot
  pmBtn.addEventListener('click', ()=>{
    try{
      const beta = parseFloat(betaEl.value), eta = parseFloat(etaEl.value);
      const Cpm = parseFloat(cpmEl.value), Ccm = parseFloat(ccmEl.value);
      const unit = document.getElementById('unit').value;
      const tmax = (eta * Math.pow(-Math.log(1e-5), 1/beta)) + 10;
      const ts = Array.from({length:300},(_,i)=> 0.1 + i*(tmax-0.1)/(299));
      const costs = ts.map(tt=> CPUT_num(tt,beta,eta,Cpm,Ccm) );
      let minIdx = 0;
      for(let i=1;i<costs.length;i++) if(costs[i]<costs[minIdx]) minIdx=i;
      const tOpt = ts[minIdx], costOpt = costs[minIdx];
      const cpmv = ts.map(tt=>CPM_num(tt,beta,eta,Cpm));
      const ccmv = ts.map(tt=>CCM_num(tt,beta,eta,Ccm));
      const traces = [
        {x:ts,y:costs,mode:'lines',name:'Total cost/hr', line:{color:'#1f77b4'}},
        {x:ts,y:cpmv,mode:'lines',name:'PM cost/hr', line:{color:'#2ca02c'}},
        {x:ts,y:ccmv,mode:'lines',name:'CM cost/hr', line:{color:'#d62728'}},
        {x:[tOpt], y:[costOpt], mode:'markers+text', text:[`T*=${tOpt.toFixed(2)}`], textposition:'top center', name:'Optimal'}
      ];
      Plotly.newPlot(plotCost, traces, {margin:{t:30}, title:`Cost per hour vs PM interval (T*=${tOpt.toFixed(2)})`});
      summary.textContent = `Optimal PM T* = ${tOpt.toFixed(2)} ${unit}. Min cost/hr = ${costOpt.toFixed(3)}`;
    }catch(err){
      summary.textContent = 'Error: '+err.message;
    }
  });

  // Export CSV
  exportCsvBtn.addEventListener('click', ()=>{
    const beta = parseFloat(betaEl.value), eta = parseFloat(etaEl.value);
    const Cpm = parseFloat(cpmEl.value), Ccm = parseFloat(ccmEl.value);
    const tmax = (eta * Math.pow(-Math.log(1e-5), 1/beta)) + 10;
    const ts = Array.from({length:300},(_,i)=> 0.1 + i*(tmax-0.1)/(299));
    const rows = ['t,CPUT,CPM,CCM'];
    for(const tt of ts){
      rows.push([tt, CPUT_num(tt,beta,eta,Cpm,Ccm), CPM_num(tt,beta,eta,Cpm), CCM_num(tt,beta,eta,Ccm)].join(','));
    }
    const blob = new Blob([rows.join('\n')], {type:'text/csv;charset=utf-8;'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'weibull_cost_grid.csv'; a.click();
    URL.revokeObjectURL(url);
  });

  clearBtn.addEventListener('click', ()=>{
    document.getElementById('summary').textContent = 'Ready.';
    plotRel.innerHTML=''; plotHaz.innerHTML=''; plotCost.innerHTML='';
  });

}); // DOMContentLoaded
