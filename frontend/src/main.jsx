import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Download, GraduationCap, Menu, X } from "lucide-react";
import { AboutPage, ComparisonPage, ExplainablePage, modelLabel, OverviewPage, PredictionPage, pretty } from "./pages.jsx";
import "./style.css";

const API=import.meta.env.DEV?(import.meta.env.VITE_API_URL||"http://127.0.0.1:8000"):"";
const ROUTES=[["Overview","/"],["Student Prediction","/prediction"],["Model Comparison","/comparison"],["Explainable AI","/explainable-ai"],["About Project","/about"]];

function App(){
  const [route,setRoute]=useState(window.location.pathname),[fields,setFields]=useState([]),[models,setModels]=useState([]),[featureSets,setFeatureSets]=useState([]),[featureColumns,setFeatureColumns]=useState({}),[ranking,setRanking]=useState([]),[analytics,setAnalytics]=useState(null);
  const [values,setValues]=useState({}),[model,setModel]=useState("RandomForest"),[featureSet,setFeatureSet]=useState("All"),[result,setResult]=useState(null),[loading,setLoading]=useState(false);
  const [error,setError]=useState(""),[apiReady,setApiReady]=useState(false),[installEvent,setInstallEvent]=useState(null),[menuOpen,setMenuOpen]=useState(false);
  const navigate=useCallback(path=>{if(!ROUTES.some(([,href])=>href===path))return; if(window.location.pathname!==path)window.history.pushState({},"",path);setRoute(path);setMenuOpen(false)},[]);
  useEffect(()=>{const pop=()=>setRoute(window.location.pathname);window.addEventListener("popstate",pop);return()=>window.removeEventListener("popstate",pop)},[]);
  useEffect(()=>{window.scrollTo(0,0)},[route]);
  useEffect(()=>{const install=e=>{e.preventDefault();setInstallEvent(e)};window.addEventListener("beforeinstallprompt",install);return()=>window.removeEventListener("beforeinstallprompt",install)},[]);
  useEffect(()=>{
    let alive=true;
    Promise.allSettled([
      fetch(`${API}/api/schema`).then(r=>{if(!r.ok)throw new Error("Could not load the student input schema.");return r.json()}),
      fetch(`${API}/api/models`).then(r=>{if(!r.ok)throw new Error("Saved model metadata is unavailable.");return r.json()}),
      fetch(`${API}/api/feature-ranking`).then(r=>r.ok?r.json():[]),
      fetch(`${API}/api/analytics`).then(r=>{if(!r.ok)throw new Error("Evaluation results are unavailable – run the saved-model evaluator.");return r.json()}),
    ]).then(([schemaResult,modelResult,rankingResult,analyticsResult])=>{
      if(!alive)return;
      if(schemaResult.status==="fulfilled"&&modelResult.status==="fulfilled"){
        const schema=schemaResult.value,modelData=modelResult.value;
        setFields(schema.fields);setValues(Object.fromEntries(schema.fields.map(f=>[f.name,String(f.default??"")])));
        setModels(modelData.models);setFeatureSets(modelData.feature_sets);setFeatureColumns(modelData.feature_columns||{});
        setModel(modelData.models.includes("RandomForest")?"RandomForest":modelData.models[0]);setFeatureSet(modelData.feature_sets.includes("All")?"All":modelData.feature_sets[0]);setApiReady(true);
      }else{setApiReady(false);setError(schemaResult.reason?.message||modelResult.reason?.message||"Backend unavailable.")}
      if(rankingResult.status==="fulfilled")setRanking(Array.isArray(rankingResult.value)?rankingResult.value.filter(x=>x.feature&&Number.isFinite(Number(x.mean_abs_shap))&&Number(x.mean_abs_shap)>=0):[]);
      if(analyticsResult.status==="fulfilled")setAnalytics(analyticsResult.value);
      else setError(previous=>previous||analyticsResult.reason?.message||"Evaluation results are unavailable.");
    });
    return()=>{alive=false};
  },[]);
  const onProfileChange=useCallback(updater=>{setValues(old=>typeof updater==="function"?updater(old):updater);setResult(null);setError("")},[]);
  const changeModel=useCallback(value=>{setModel(value);setResult(null)},[]),changeFeatureSet=useCallback(value=>{setFeatureSet(value);setResult(null)},[]);
  function resetForm(){setValues(Object.fromEntries(fields.map(f=>[f.name,String(f.default??"")])));setResult(null);setError("")}
  async function predict(event){
    event.preventDefault();setLoading(true);setError("");setResult(null);
    const studentData={};
    for(const field of fields){const raw=values[field.name];if(raw===""||raw==null){setError(`${pretty(field.name)} is required.`);setLoading(false);return}if(field.kind==="number"){const n=Number(raw);if(!Number.isFinite(n)||n<field.min||n>field.max){setError(`${pretty(field.name)} must be between ${field.min} and ${field.max}.`);setLoading(false);return}studentData[field.name]=n}else studentData[field.name]=raw}
    try{const response=await fetch(`${API}/api/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({student_data:studentData,model_name:model,feature_set:featureSet})});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:"Prediction failed. Check the supplied details.");const p=Number(data.placement_probability);if(!Number.isFinite(p)||p<0||p>1||![0,1].includes(data.predicted_class))throw new Error("The API returned an invalid prediction.");setResult(data)}catch(e){setError(e.message||"Could not reach the prediction API.")}finally{setLoading(false)}
  }
  async function installApp(){if(installEvent){await installEvent.prompt();setInstallEvent(null)}else setError("To install on iPhone, open this page in Safari and choose Share → Add to Home Screen.")}

  const page=useMemo(()=>{
    if(route==="/")return <OverviewPage analytics={analytics} models={models} metrics={analytics?.metrics||[]} ranking={ranking} selectedCount={featureColumns[featureSet]?.length??null} onNavigate={navigate}/>;
    if(route==="/prediction")return <PredictionPage fields={fields} fieldColumns={featureColumns} values={values} setValues={onProfileChange} models={models} featureSets={featureSets} model={model} setModel={changeModel} featureSet={featureSet} setFeatureSet={changeFeatureSet} result={result} loading={loading} apiReady={apiReady} onSubmit={predict} onReset={resetForm}/>;
    if(route==="/comparison")return <ComparisonPage analytics={analytics} models={models} featureSets={featureSets}/>;
    if(route==="/explainable-ai")return <ExplainablePage ranking={ranking} analytics={analytics} models={models} featureSets={featureSets} featureColumns={featureColumns}/>;
    if(route==="/about")return <AboutPage analytics={analytics} models={models}/>;
    return <section className="panel empty-data"><h1>Page not found</h1><button className="text-link" onClick={()=>navigate("/")}>Return to Overview</button></section>;
  },[route,analytics,models,ranking,featureColumns,featureSet,navigate,fields,values,onProfileChange,featureSets,model,changeModel,changeFeatureSet,result,loading,apiReady]);

  return <div className="app-shell"><header className="topbar"><a className="brand" href="/" onClick={e=>{e.preventDefault();navigate("/")}} aria-label="CampusPath Overview"><div className="brand-icon"><GraduationCap size={24}/></div><div><strong>Campus<span>Path</span></strong><small>STUDENT PLACEMENT ANALYTICS</small></div></a>
    <nav className={menuOpen?"nav open":"nav"} aria-label="Main navigation">{ROUTES.map(([label,path])=><a key={path} href={path} className={route===path?"active":""} aria-current={route===path?"page":undefined} onClick={e=>{e.preventDefault();navigate(path)}}>{label}</a>)}</nav>
    <div className="header-actions"><div className={`api-status ${apiReady?"online":"offline"}`} role="status"><i/>{apiReady?"Backend connected":"Backend unavailable"}</div>{"serviceWorker" in navigator&&<button className="install-button" type="button" onClick={installApp}><Download size={15}/> Install app</button>}<button className="menu-button" aria-label={menuOpen?"Close navigation":"Open navigation"} aria-expanded={menuOpen} onClick={()=>setMenuOpen(!menuOpen)}>{menuOpen?<X/>:<Menu/>}</button></div>
  </header><main className="route-main"><div className="route-transition" key={route}>{error&&<div className="alert" role="alert">{error}</div>}{page}</div><footer><span>CAMPUSPATH · STUDENT PLACEMENT ANALYTICS</span><span>Educational use only · Model estimates are not guarantees of employment.</span></footer></main></div>;
}
createRoot(document.getElementById("root")).render(<App/>);
