from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT))
from src.statistics import clustered_bootstrap  # noqa: E402
from src.unified_latent_sampler import CANDIDATE_LAYERS, LATENT_CONDITIONS, MATURITY_EPSILON  # noqa: E402

N_BINS = 5


def load() -> tuple[list[dict], list[dict]]:
    paths=sorted((PROJECT/"traces/unified_latent_state_probe/formal").glob("sample_*.json"))
    if len(paths)!=200: raise RuntimeError(f"Expected 200 formal samples, found {len(paths)}")
    samples=[json.loads(path.read_text()) for path in paths]
    if [s["sample_id"] for s in samples]!=list(range(200)) or not all(s["sanity"]["passed"] for s in samples): raise RuntimeError("Formal samples incomplete or failed")
    return samples,[r for s in samples for r in s["observations"]]


def metric(values,clusters,seed:int)->dict:
    values=np.asarray(values,dtype=float);clusters=np.asarray(clusters);valid=np.isfinite(values);values,clusters=values[valid],clusters[valid]
    if not len(values): return {"n":0,"mean":None,"median":None,"percent_positive":None,"sample_cluster_bootstrap_95_ci":[None,None]}
    lo,hi=clustered_bootstrap(values,clusters,seed=seed)
    return {"n":int(len(values)),"mean":float(values.mean()),"median":float(np.median(values)),"percent_positive":float(100*np.mean(values>0)),"sample_cluster_bootstrap_95_ci":[lo,hi]}


def condition_summary(records:list[dict],seed:int)->dict:
    clusters=[r["sample_id"] for r in records]; result={}
    for li,layer in enumerate(CANDIDATE_LAYERS):
        layer_item={}
        for ci,condition in enumerate(LATENT_CONDITIONS):
            rows=[r["layers"][str(layer)][condition] for r in records]
            item={
                "self_delta_logp":metric([x["self_delta_logp"] for x in rows],clusters,seed+li*100+ci*3),
                "downstream_gain":metric([x["downstream_gain"] for x in rows],clusters,seed+li*100+ci*3+1),
                "hit_rate":float(np.mean([x["hit"] for x in rows])),"w2r":int(sum(x["w2r"] for x in rows)),"r2w":int(sum(x["r2w"] for x in rows)),
            };item["net_gain"]=item["w2r"]-item["r2w"];layer_item[condition]=item
        comparisons={
            "previous_minus_early":("previous","early"),"previous_minus_shuffle":("previous","shuffle"),"previous_minus_random":("previous","random"),
            "endpoint_minus_previous":("endpoint","previous"),"endpoint_minus_random":("endpoint","random"),
        };layer_item["comparisons"]={}
        for ci,(name,(left,right)) in enumerate(comparisons.items()):
            layer_item["comparisons"][name]={}
            for fi,field in enumerate(("self_delta_logp","downstream_gain")):
                values=[r["layers"][str(layer)][left][field]-r["layers"][str(layer)][right][field] for r in records]
                layer_item["comparisons"][name][field]=metric(values,clusters,seed+li*100+50+ci*3+fi)
        result[str(layer)]=layer_item
    result["hard"]={
        "downstream_gain":metric([r["hard_downstream_gain"] for r in records],clusters,seed+1000),
        "future_consistent":metric([r["hard_downstream_gain"] for r in records if r["hard_future_token_consistent"]],[r["sample_id"] for r in records if r["hard_future_token_consistent"]],seed+1001),
        "future_inconsistent":metric([r["hard_downstream_gain"] for r in records if not r["hard_future_token_consistent"]],[r["sample_id"] for r in records if not r["hard_future_token_consistent"]],seed+1002),
    }
    return result


def assign_bins(records:list[dict])->list[dict]:
    order=np.argsort([r["maturity"] for r in records],kind="stable"); groups=np.array_split(order,N_BINS); bins=[]
    for index,indices in enumerate(groups):
        subset=[records[int(i)] for i in indices]; values=[r["maturity"] for r in subset]; entropy=[r["entropy"] for r in subset]
        bins.append({"index":index,"records":subset,"n":len(subset),"maturity_range":[float(min(values)),float(max(values))],"entropy_range":[float(min(entropy)),float(max(entropy))]})
    return bins


def binned_summary(bins:list[dict],seed:int)->list[dict]:
    result=[]
    for bi,bin_item in enumerate(bins):
        item={k:v for k,v in bin_item.items() if k!="records"}; item["layers"]={}; records=bin_item["records"];clusters=[r["sample_id"] for r in records]
        for li,layer in enumerate(CANDIDATE_LAYERS):
            layer_item={}
            for condition in ("previous","endpoint","early","shuffle","random"):
                layer_item[condition]={
                    "self_delta_logp":metric([r["layers"][str(layer)][condition]["self_delta_logp"] for r in records],clusters,seed+bi*1000+li*100+LATENT_CONDITIONS.index(condition)*2),
                    "downstream_gain":metric([r["layers"][str(layer)][condition]["downstream_gain"] for r in records],clusters,seed+bi*1000+li*100+LATENT_CONDITIONS.index(condition)*2+1),
                }
            item["layers"][str(layer)]=layer_item
        item["hard_downstream_gain"]=metric([r["hard_downstream_gain"] for r in records],clusters,seed+bi*1000+900)
        result.append(item)
    return result


def continuous_associations(records:list[dict])->dict:
    variables=("maturity","entropy","relative_entropy","top1_probability","top1_top2_margin"); result={}
    for layer in CANDIDATE_LAYERS:
        item={}
        for condition in ("previous","endpoint"):
            item[condition]={}
            for field in ("self_delta_logp","downstream_gain"):
                outcome=np.array([r["layers"][str(layer)][condition][field] for r in records])
                item[condition][field]={}
                for variable in variables:
                    x=np.array([r[variable] for r in records])
                    if np.ptp(x)==0 or np.ptp(outcome)==0:
                        pearson=spearman=None
                    else:
                        pearson=float(pearsonr(x,outcome).statistic)
                        spearman=float(spearmanr(x,outcome).statistic)
                    item[condition][field][variable]={"pearson":pearson,"spearman":spearman}
        result[str(layer)]=item
    return result


def reversibility(records:list[dict])->dict:
    groups={}
    for r in records: groups.setdefault((r["sample_id"],r["block_index"],r["absolute_position"]),[]).append(r)
    entropy_up=maturity_down=total=0; positions_multiple=0
    for rows in groups.values():
        rows.sort(key=lambda r:r["step_in_block"])
        if len(rows)>1: positions_multiple+=1
        for left,right in zip(rows,rows[1:]):
            total+=1;entropy_up+=right["entropy"]>left["entropy"];maturity_down+=right["maturity"]<left["maturity"]
    return {"positions_with_multiple_probes":positions_multiple,"transitions":total,"entropy_increase_frequency":entropy_up/total if total else None,"maturity_decrease_frequency":maturity_down/total if total else None}


def representative_layer(overall:dict)->int|None:
    qualified=[]
    for layer in CANDIDATE_LAYERS:
        item=overall[str(layer)]; prev=item["previous"]
        if prev["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]<=0 or prev["downstream_gain"]["sample_cluster_bootstrap_95_ci"][0]<=0: continue
        if all(item["comparisons"][f"previous_minus_{control}"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0 for control in ("early","shuffle","random")):
            qualified.append(layer)
    if len(qualified)!=1:return None
    return qualified[0]


def preference_by_bin(bins:list[dict],representative:int|None,seed:int)->list[dict]:
    result=[]
    for bi,bin_item in enumerate(bins):
        records=bin_item["records"];clusters=[r["sample_id"] for r in records]
        states={"MASK":[0.0]*len(records),"HARD":[r["hard_downstream_gain"] for r in records]}
        if representative is not None: states["LATENT"]=[r["layers"][str(representative)]["previous"]["downstream_gain"] for r in records]
        comparisons={}; names=list(states)
        for i,left in enumerate(names):
            for right in names[i+1:]: comparisons[f"{left}_minus_{right}"]=metric(np.array(states[left])-np.array(states[right]),clusters,seed+bi*100+i*10+names.index(right))
        means={name:float(np.mean(values)) for name,values in states.items()};winner=max(means,key=means.get);supported=True
        for other in states:
            if other==winner:continue
            key=f"{winner}_minus_{other}" if f"{winner}_minus_{other}" in comparisons else f"{other}_minus_{winner}"
            ci=comparisons[key]["sample_cluster_bootstrap_95_ci"]
            if key.startswith(winner+"_minus_"): supported&=ci[0]>0
            else:supported&=ci[1]<0
        result.append({"bin":bi,"maturity_range":bin_item["maturity_range"],"means":means,"comparisons":comparisons,"preferred":winner if supported else "TIE"})
    return result


def _classify_decision(training_free:bool,endpoint_usable:bool,previous_usable:bool,maturity_structured:bool)->dict:
    if training_free:return {"code":"A","title":"TRAINING-FREE LATENT STATE SIGNAL"}
    if endpoint_usable and not previous_usable:return {"code":"B","title":"ENDPOINT INTERFACE EXISTS, SIMPLE CARRY FAILS"}
    if endpoint_usable and previous_usable:return {"code":"E","title":"PARTIAL CARRY SIGNAL, NOT CONTROL-ROBUST"}
    if maturity_structured:return {"code":"C","title":"ENTROPY STRUCTURE EXISTS, BUT NO LATENT UTILITY"}
    return {"code":"D","title":"NO USABLE LATENT INTERFACE"}


def decide(overall:dict,binned:list[dict],preferences:list[dict],associations:dict,representative:int|None)->dict:
    # All A criteria must refer to the same preselected representative layer.
    # representative_layer() already requires positive overall self/downstream
    # effects and superiority to every control at that layer.
    positive_bins=[] if representative is None else [
        b["index"] for b in binned
        if b["layers"][str(representative)]["previous"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0
    ]
    latent_preferred=representative is not None and any(p["preferred"]=="LATENT" for p in preferences)
    training_free=bool(positive_bins and latent_preferred)
    endpoint_usable=any(
        overall[str(layer)]["endpoint"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0
        or overall[str(layer)]["endpoint"]["downstream_gain"]["sample_cluster_bootstrap_95_ci"][0]>0 for layer in CANDIDATE_LAYERS
    )
    previous_usable=any(overall[str(layer)]["previous"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0 for layer in CANDIDATE_LAYERS)
    maturity_structured=max(abs(associations[str(layer)]["previous"]["self_delta_logp"]["maturity"]["spearman"]) for layer in CANDIDATE_LAYERS)>.10
    return _classify_decision(training_free,endpoint_usable,previous_usable,maturity_structured)


def _number(text:str,reference:bool=False)->str|None:
    patterns=[r"\\boxed\{\s*([^{}]+)\s*\}",r"####\s*([^\n]+)"] if not reference else [r"####\s*([^\n]+)"]
    for pattern in patterns:
        found=re.findall(pattern,text)
        if found:
            value=re.sub(r"[^0-9.\-]","",found[-1].replace(",",""))
            if value:return value
    found=re.findall(r"-?\d[\d,]*(?:\.\d+)?",text);return found[-1].replace(",","") if found else None


def make_figures(overall,binned,preferences,representative)->None:
    directory=PROJECT/"figures/unified_latent_state_probe";directory.mkdir(parents=True,exist_ok=True);layers=list(CANDIDATE_LAYERS)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for condition in LATENT_CONDITIONS:
        means=[overall[str(l)][condition]["self_delta_logp"]["mean"] for l in layers];cis=[overall[str(l)][condition]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"] for l in layers]
        ax.errorbar(layers,means,yerr=[[m-c[0] for m,c in zip(means,cis)],[c[1]-m for m,c in zip(means,cis)]],marker="o",capsize=3,label=condition.title())
    ax.axhline(0,color="black",lw=.7);ax.set_xlabel("Intervention layer");ax.set_ylabel("Self DeltaLogP");ax.legend();fig.tight_layout();fig.savefig(directory/"01_layer_self_delta.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(len(layers));w=.25
    w2r=[overall[str(l)]["previous"]["w2r"] for l in layers];r2w=[overall[str(l)]["previous"]["r2w"] for l in layers];net=[overall[str(l)]["previous"]["net_gain"] for l in layers]
    ax.bar(x-w,w2r,w,label="W2R");ax.bar(x,r2w,w,label="R2W");ax.bar(x+w,net,w,label="Net");ax.set_xticks(x,layers);ax.set_xlabel("Layer");ax.legend();fig.tight_layout();fig.savefig(directory/"02_previous_transitions.png",dpi=180);plt.close(fig)
    centers=[np.mean(b["maturity_range"]) for b in binned]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for layer in layers:ax.plot(centers,[b["layers"][str(layer)]["previous"]["self_delta_logp"]["mean"] for b in binned],marker="o",label=str(layer))
    ax.axhline(0,color="black",lw=.7);ax.set_xlabel("Maturity quantile center");ax.set_ylabel("Previous SelfDeltaLogP");ax.legend(title="Layer");fig.tight_layout();fig.savefig(directory/"03_maturity_previous_curves.png",dpi=180);plt.close(fig)
    for filename,condition,title in (("04_previous_heatmap.png","previous","Previous Carry"),("05_endpoint_heatmap.png","endpoint","Endpoint Oracle")):
        matrix=np.array([[b["layers"][str(layer)][condition]["self_delta_logp"]["mean"] for b in binned] for layer in layers]);fig,ax=plt.subplots(figsize=(8,4.5));im=ax.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=-np.max(abs(matrix)),vmax=np.max(abs(matrix)));ax.set_yticks(range(len(layers)),layers);ax.set_xticks(range(N_BINS),[f"Q{i+1}" for i in range(N_BINS)]);ax.set_ylabel("Layer");ax.set_xlabel("Maturity quantile");ax.set_title(title+" SelfDeltaLogP");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(directory/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.axhline(0,color="black",label="MASK")
    for layer in layers:ax.plot(range(N_BINS),[b["layers"][str(layer)]["previous"]["downstream_gain"]["mean"] for b in binned],marker="o",alpha=.65,label=f"LATENT L{layer}")
    ax.plot(range(N_BINS),[b["hard_downstream_gain"]["mean"] for b in binned],marker="s",lw=2,label="HARD");ax.set_xticks(range(N_BINS),[f"Q{i+1}" for i in range(N_BINS)]);ax.set_ylabel("DownstreamGain");ax.set_xlabel("Maturity quantile");ax.legend(ncol=2);fig.tight_layout();fig.savefig(directory/"06_mask_latent_hard.png",dpi=180);plt.close(fig)
    colors={"MASK":"#4c78a8","LATENT":"#59a14f","HARD":"#f28e2b","TIE":"#bab0ac"};fig,ax=plt.subplots(figsize=(8,2.8));values=[p["preferred"] for p in preferences]
    for i,value in enumerate(values):ax.bar(i,1,color=colors[value]);ax.text(i,.5,value,ha="center",va="center",fontweight="bold")
    ax.set_xticks(range(N_BINS),[f"Q{i+1}" for i in range(N_BINS)]);ax.set_yticks([]);ax.set_xlabel("Maturity quantile");ax.set_title(f"Preferred downstream state (representative latent={representative})");fig.tight_layout();fig.savefig(directory/"07_preferred_state.png",dpi=180);plt.close(fig)


def main()->None:
    samples,records=load();bins=assign_bins(records);overall=condition_summary(records,20285000);binned=binned_summary(bins,20287000);associations=continuous_associations(records);reversal=reversibility(records);representative=representative_layer(overall);preferences=preference_by_bin(bins,representative,20290000);decision=decide(overall,binned,preferences,associations,representative)
    correct=sum(_number(s["decoded_output"])==_number(s["reference_answer"],True) for s in samples);environment=json.loads((PROJECT/"results/environment.json").read_text())
    cohort_diagnostics={
        "strict_future_endpoint":all(r["endpoint_horizon"]>0 for r in records),
        "endpoint_horizon_range":[min(r["endpoint_horizon"] for r in records),max(r["endpoint_horizon"] for r in records)],
        "downstream_count_range":[min(r["downstream_count"] for r in records),max(r["downstream_count"] for r in records)],
        "mean_maturity_by_progress":{
            progress:float(np.mean([r["maturity"] for r in records if r["progress"]==progress]))
            for progress in sorted({r["progress"] for r in records})
        },
        "multiplicity_note":"Layer/bin searches are exploratory and use unadjusted 95% clustered-bootstrap intervals; individual positive cells are not family-wise-error controlled.",
    }
    answers={
        "Q1_recent_state_consumable":any(overall[str(l)]["previous"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0 for l in CANDIDATE_LAYERS),
        "Q2_recent_beats_controls":any(all(overall[str(l)]["comparisons"][f"previous_minus_{c}"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0 for c in ("early","shuffle","random")) for l in CANDIDATE_LAYERS),
        "Q3_endpoint_causally_useful":any(overall[str(l)]["endpoint"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0 or overall[str(l)]["endpoint"]["downstream_gain"]["sample_cluster_bootstrap_95_ci"][0]>0 for l in CANDIDATE_LAYERS),
        "Q4_headroom_layer":max(CANDIDATE_LAYERS,key=lambda l:overall[str(l)]["endpoint"]["self_delta_logp"]["mean"]),
        "Q5_maturity_dependence":{str(l):associations[str(l)]["previous"]["self_delta_logp"]["maturity"] for l in CANDIDATE_LAYERS},
        "Q6_positive_layer_maturity_cells":[{"layer":l,"bin":b["index"]} for l in CANDIDATE_LAYERS for b in binned if b["layers"][str(l)]["previous"]["self_delta_logp"]["sample_cluster_bootstrap_95_ci"][0]>0],
        "Q7_preferences":[p["preferred"] for p in preferences],"Q8_uncertainty_reversible":reversal["maturity_decrease_frequency"]>0.05,
    }
    summary={"status":"complete","samples":200,"observations":len(records),"all_official_outputs_equal":all(s["sanity"]["reference_equals_unified_traced"] for s in samples),"candidate_layers":list(CANDIDATE_LAYERS),"maturity_epsilon":MATURITY_EPSILON,"maturity_bins":[{k:v for k,v in b.items() if k!="records"} for b in bins],"overall":overall,"by_maturity_bin":binned,"continuous_associations":associations,"reversibility":reversal,"cohort_diagnostics":cohort_diagnostics,"representative_latent_layer":representative,"state_preferences":preferences,"final_correct_samples":correct,"scientific_answers":answers,"decision":decision,"environment":environment}
    (PROJECT/"results/unified_latent_state_probe_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["UNIFIED LAYER × ENTROPY × STATE PROBE","="*39,"",f"DECISION: {decision['code']}. {decision['title']}",f"Samples: 200, observations: {len(records)}, trajectory parity: {summary['all_official_outputs_equal']}",f"Candidate layers: {list(CANDIDATE_LAYERS)}; maturity bins: {N_BINS}; epsilon: {MATURITY_EPSILON}","","COHORT AND INFERENCE DIAGNOSTICS",json.dumps(cohort_diagnostics,indent=2),"", "SCIENTIFIC QUESTIONS",json.dumps(answers,indent=2),"","OVERALL LAYER RESULTS",json.dumps(overall,indent=2),"","MATURITY BINS",json.dumps(summary["maturity_bins"],indent=2),"","LAYER × MATURITY",json.dumps(binned,indent=2),"","CONTINUOUS ASSOCIATIONS",json.dumps(associations,indent=2),"","MASK/LATENT/HARD PREFERENCES",json.dumps(preferences,indent=2),"","HARD SEMANTIC STRATIFICATION",json.dumps(overall["hard"],indent=2),"","TEMPORAL REVERSIBILITY",json.dumps(reversal,indent=2),"","ENVIRONMENT",json.dumps(environment,indent=2)]
    (PROJECT/"results/report_unified_latent_state_probe.txt").write_text("\n".join(lines)+"\n");make_figures(overall,binned,preferences,representative)


if __name__=="__main__":main()
