"""Inert public verifier: sanitized arithmetic, not proof of private execution.

Import only defines functions/constants. CLI reads local evidence and its optional adjacent pinned input-scope audit and
prints tables; no mutation, environment, network, subprocess or product code.
"""
import argparse
import hashlib
from pathlib import Path
import json
import math
import statistics

REVISIONS=('315c10242ddf7a1cec3bccbf550a942320e09557','d0b6a14a4f317fea3c1e117f627289aad1b1c964')
WORK=('offered','scored','characters','tokens','padded_tokens','batches','encode_us','forward_us')
COST=('controller_observed_native_wall_seconds','engine_open_wall_us','cumulative_user_seconds_through_last_search','cumulative_system_seconds_through_last_search','native_sampled_peak_rss_kib','native_observed_hwm_kib','owned_pg_main_pid_sampled_peak_rss_kib','full_lifetime_cpu_seconds','total_service_memory_kib')
CORRECT=('same_final_actual_inputs','same_final_raw_bits_and_order','all_changed_and_hot_exact','legacy_main_context_failure','retrieval_or_input_context_mismatch','first_fresh_and_new_cross_source_exact')
ATTEST=('private_complete_80_processes_880_calls','private_bound_source_build_provider_and_raw_rows_passed','private_fresh_independent_oracles_passed','private_off_cross_source_exact','private_full_payload_reloaded_endpoint_reconstruction_passed','private_allowed_source_derived_metadata_writes_only','private_original_stopped_seed_unchanged')
CONDITIONS=('entrypoint','access_mode','profile','provider','precision','document_count','tick_hz','provider_nodes','embedding_model_revision','reranker_model_revision','runtime_version','cpu_model','architecture','cpu_quota','source_order_by_block','cold_scope','memory_scope')
DISK=(('index','logical_bytes'),('index','allocated_bytes'),('app_executable','logical_bytes'),('app_executable','allocated_bytes'),('shared_model_runtime_assets','logical_bytes'),('shared_model_runtime_assets','allocated_bytes'),('temporary_clones','peak_allocated_bytes'))


def require(ok,message):
    if not ok:raise ValueError(message)


def keys(value,expected):
    require(type(value) is dict and set(value)==set(expected),'unknown/missing object fields')


def number(value,nullable=False,integer=False):
    if value is None and nullable:return
    require(type(value) is int if integer else type(value) in (int,float),'numeric type (bool refused)')
    require(math.isfinite(value) and value>=0,'nonnegative finite numeric value')


def strict_json(text):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key')
            result[key]=value
        return result
    def nonfinite(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)


def matrix():
    return {(block,arm,tier,limit,scenario,initial) for block in range(4) for arm in ('main','stack')
            for tier,limit,scenario,initial in
            [('accurate',l,s,i) for l in (5,10) for s in ('scenario_1','scenario_2') for i in ('A','B')]+[('off',l,'scenario_1','A') for l in (5,10)]}


def phase(tier,step):
    if tier=='off':return ('off-cold','off-hot','off-new-query')[step]
    return 'cold' if step==0 else 'initial-hot' if step<6 else 'changed-context' if step==6 else 'changed-hot' if step<12 else 'new-query'


def validate(data):
    keys(data,('schema_version','comparison','conditions','processes','disk','attestations'))
    require(type(data['schema_version']) is int and data['schema_version']==1,'schema version')
    comparison=data['comparison'];keys(comparison,('before_commit','after_commit','process_count','call_count'))
    require((comparison['before_commit'],comparison['after_commit'])==REVISIONS,'compared revisions')
    require(type(comparison['process_count']) is int and comparison['process_count']==80 and type(comparison['call_count']) is int and comparison['call_count']==880,'80/880 declared count')
    conditions=data['conditions'];keys(conditions,CONDITIONS)
    for key,expected in [('entrypoint','Engine.search_reranked'),('access_mode','ReadOnly'),('profile','accuracy'),('provider','CPUExecutionProvider')]:
        require(conditions[key]==expected,'fixed execution condition')
    require(type(conditions['document_count']) is int and conditions['document_count']==230 and type(conditions['tick_hz']) is int and conditions['tick_hz']==100,'document/tick scope')
    keys(conditions['provider_nodes'],('embedding','accurate_reranker'))
    require(all(type(v) is int for v in conditions['provider_nodes'].values()) and conditions['provider_nodes']=={'embedding':1023,'accurate_reranker':295},'provider node role')
    require(conditions['source_order_by_block']==[['main','stack'],['stack','main'],['stack','main'],['main','stack']],'rotated block order')
    for key in ('precision','embedding_model_revision','reranker_model_revision','runtime_version','cpu_model','architecture','cpu_quota'):
        value=conditions[key]
        require(value is None or (type(value) is str and 0<len(value)<=200 and value==value.strip() and '|' not in value and all(c.isprintable() for c in value) and not any(c in value for c in '\\/:@\n\r\t')),'sanitized condition string')
    require(conditions['cold_scope']=='fresh process and session; OS file cache warmed by preflight','cold scope')
    require(conditions['memory_scope']=='native process RSS and HWM; owned PG main PID sampled separately; total service N/A','memory scope')
    att=data['attestations'];keys(att,ATTEST+('validation_scope',))
    require(all(type(att[key]) is bool and att[key] for key in ATTEST),'private-validation attestation required')
    require(att['validation_scope']=='private_attestation_not_public_independent_proof','limited attestation scope')
    require(type(data['processes']) is list and len(data['processes'])==80,'80 process records')
    seen=set();calls=0
    for p in data['processes']:
        keys(p,('block','arm','tier','limit','scenario','initial_context','calls','correctness','cost'))
        require(type(p['block']) is int and type(p['limit']) is int,'matrix integer types')
        cell=tuple(p[k] for k in ('block','arm','tier','limit','scenario','initial_context'))
        require(cell in matrix() and cell not in seen,'unexpected/duplicate process cell');seen.add(cell)
        accurate=p['tier']=='accurate';count=13 if accurate else 3
        require(type(p['calls']) is list and len(p['calls'])==count,'phase cardinality')
        calls+=count
        for step,c in enumerate(p['calls']):
            keys(c,('step','phase','context','wall_us','cpu_user_ticks','cpu_system_ticks','rss_kib','hwm_kib','quality','work'))
            context=p['initial_context'] if step<6 else ('B' if p['initial_context']=='A' else 'A') if step<12 else 'N'
            if not accurate:context='A' if step<2 else 'N'
            require(type(c['step']) is int and c['step']==step and c['phase']==phase(p['tier'],step).replace('-','_') and c['context']==context,'phase/context chronology')
            for k in ('wall_us','cpu_user_ticks','cpu_system_ticks','rss_kib','hwm_kib'):number(c[k],integer=True)
            require(c['hwm_kib']>=c['rss_kib'],'HWM below RSS')
            keys(c['quality'],('recall','mrr','ndcg'))
            for value in c['quality'].values():number(value);require(value<=1,'quality range')
            keys(c['work'],WORK)
            for value in c['work'].values():number(value,integer=True)
            w=c['work'];require(w['scored']<=w['offered'] and w['tokens']<=w['padded_tokens'] and w['batches']<=w['scored'],'work coherence')
            if not accurate:require(all(v==0 for v in w.values()),'Off model work')
            else:
                require(w['offered']==30,'actual offered budget')
                if step in (0,12):require(w['scored']==30 and w['batches']>0,'fresh/new successful scoring')
                if step in (*range(1,6),*range(7,12)):require(all(w[k]==0 for k in WORK if k!='offered'),'same-context hot work')
            if w['scored']==0:require(all(w[k]==0 for k in ('characters','tokens','padded_tokens','batches','forward_us')),'cost without scoring')
            else:require(all(w[k]>0 for k in ('characters','tokens','batches')),'scoring without successful work')
        correction=p['correctness']
        if not accurate:require(correction is None,'Off correctness schema')
        else:
            keys(correction,CORRECT);require(all(type(v) is bool for v in correction.values()),'correctness boolean fields')
            inp=correction['same_final_actual_inputs'];exact=correction['same_final_raw_bits_and_order'];hot=correction['all_changed_and_hot_exact']
            require(not exact or inp,'exact requires matching input')
            require(not hot or exact,'all hot exact includes changed call')
            require(correction['retrieval_or_input_context_mismatch']==(not inp),'input mismatch consistency')
            require(correction['legacy_main_context_failure']==(p['arm']=='main' and inp and not exact),'legacy context consistency')
            require(correction['first_fresh_and_new_cross_source_exact'],'fresh/new parity attestation')
            if p['arm']=='stack':require(inp and exact and hot,'stack fresh-oracle correctness')
        keys(p['cost'],COST)
        for key,value in p['cost'].items():number(value,nullable=True)
        native=p['cost']
        if native['native_sampled_peak_rss_kib'] is not None and native['native_observed_hwm_kib'] is not None:
            require(native['native_observed_hwm_kib']>=native['native_sampled_peak_rss_kib'],'process native HWM below sampled RSS')
        require(p['cost']['full_lifetime_cpu_seconds'] is None and p['cost']['total_service_memory_kib'] is None,'unmeasured full lifetime/service scope')
    require(seen==matrix() and calls==880,'complete actual matrix')
    require(type(data['disk']) is list and len(data['disk'])==len(DISK),'disk cardinality')
    seen_disk=set()
    for d in data['disk']:
        keys(d,('scope','measure','before','after'));pair=(d['scope'],d['measure'])
        require(pair in DISK and pair not in seen_disk,'unknown/duplicate disk scope');seen_disk.add(pair)
        for k in ('before','after'):number(d[k],nullable=True,integer=True)
        if pair in (('index','logical_bytes'),('app_executable','logical_bytes'),('shared_model_runtime_assets','logical_bytes')):
            require(d['before'] is not None and d['after'] is not None,'measured logical disk values required')
        if pair in (('index','logical_bytes'),('shared_model_runtime_assets','logical_bytes')):
            require(d['before']==d['after'],'shared assets and validated index logical sizes must agree')
        if pair in (('app_executable','allocated_bytes'),('shared_model_runtime_assets','allocated_bytes'),('temporary_clones','peak_allocated_bytes')):
            require(d['before'] is None and d['after'] is None,'unmeasured disk scope')
        if pair==('index','allocated_bytes'):require(d['after'] is None,'index endpoint allocation unmeasured')
    return data


def delta(before,after):
    return {'before':before,'after':after,'absolute_difference':after-before if before is not None and after is not None else None,
            'percentage_change':100*(after-before)/before if before is not None and after is not None and before!=0 else None}


def quantile(values,p):
    values=sorted(values);position=(len(values)-1)*p;lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def stability(changes,eligible,same_work,medians):
    median_deltas=[b-a for a,b in zip(medians['main'],medians['stack'])]
    spans={s:(max(v)-min(v))/statistics.median(v) if statistics.median(v) else None for s,v in medians.items()}
    meaningful=all(v is not None for v in changes)
    sign_flip=min(median_deltas)<0<max(median_deltas) or (meaningful and min(changes)<0<max(changes))
    spread=max(changes)-min(changes) if meaningful else None
    return {'arm_four_block_medians':medians,'arm_span_over_median':spans,'four_block_percentage_changes':changes,
            'paired_block_median_absolute_deltas':median_deltas,'sign_flip':sign_flip,'spread_percentage_points':spread,
            'stable_claim_eligible':bool(eligible and same_work and meaningful and not sign_flip and all(v is not None and v<=.10 for v in spans.values()) and spread<=10)}


AUDIT_CANONICAL_SHA256='75a3a959a8ff6af9dad69d6e8271bce7e6766be43b171db342cb8052dbc2f3a8'


def input_scope(data, audit):
    if audit is None:
        return {} # No input-change attestation: no changed-input eligibility.
    encoded=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    require(hashlib.sha256(encoded(audit)).hexdigest()==AUDIT_CANONICAL_SHA256, 'input audit identity differs')
    require(hashlib.sha256(encoded(data)).hexdigest()==audit['evidence_canonical_sha256'], 'input audit evidence binding differs')
    return audit['scenario_changed_inputs']


def tables(data, audit=None):
    validate(data);scopes=input_scope(data,audit);groups={};processes=data['processes'];rows=[]
    for p in processes:
        for c in p['calls']:
            group=(p['tier'],p['limit'],p['scenario'],p['initial_context'],c['phase'].replace('_','-'),c['context'])
            groups.setdefault(group,[]).append((p,c))
    for group,items in sorted(groups.items()):
        eligible=bool(scopes.get(group[2],False)) and all(p['correctness']['same_final_raw_bits_and_order'] and p['correctness']['all_changed_and_hot_exact'] for p,c in items) if group[4] in ('changed-context','changed-hot') else True
        ordered={s:sorted([(p['block'],c['step'],c['work']) for p,c in items if p['arm']==s],key=lambda v:v[:2]) for s in ('main','stack')}
        same_work=all([tuple(w[k] for k in WORK[:6]) for _,_,w in ordered[s]]==[tuple(w[k] for k in WORK[:6]) for _,_,w in ordered['main']] for s in ('main','stack'))
        definitions=[('quality_'+k,lambda c,k=k:c['quality'][k]) for k in ('recall','mrr','ndcg')]
        definitions += [('wall_us',lambda c:c['wall_us']),('cpu_user_seconds',lambda c:c['cpu_user_ticks']/100),('cpu_system_seconds',lambda c:c['cpu_system_ticks']/100),('rss_kib',lambda c:c['rss_kib']),('hwm_kib',lambda c:c['hwm_kib'])]
        definitions += [(k,lambda c,k=k:c['work'][k]) for k in WORK[1:]]
        for metric,extract in definitions:
            for statistic,percent in [('p50',.5),('p95_descriptive',.95)] if metric=='wall_us' else [('mean',None)]:
                reduce=lambda v:quantile(v,percent) if percent is not None else statistics.mean(v)
                values={s:reduce([extract(c) for p,c in items if p['arm']==s]) for s in ('main','stack')}
                medians={s:[] for s in ('main','stack')};blocks=[];changes=[]
                for block in range(4):
                    blockvalues={s:[extract(c) for p,c in items if p['arm']==s and p['block']==block] for s in ('main','stack')}
                    for s,v in blockvalues.items():medians[s].append(statistics.median(v))
                    result=delta(reduce(blockvalues['main']),reduce(blockvalues['stack']));blocks.append(result);changes.append(result['percentage_change'])
                rows.append({'configuration':list(group),'metric':metric,'statistic':statistic,'samples_per_arm':len(items)//2,'accuracy_eligible':eligible,'history_scope':('changed-input history' if scopes.get(group[2],False) else 'unchanged-input control' if group[2] in scopes else 'input-change scope unproven') if group[4] in ('changed-context','changed-hot') else 'other phase','same_successful_work':same_work,
                             **delta(values['main'],values['stack']),**stability(changes,eligible,same_work,medians),'four_block_statistics':blocks})
    correctness=[];control_diagnostics=[]
    for limit in (5,10):
        for scenario in ('scenario_1','scenario_2'):
            for initial in ('A','B'):
                items=[p for p in processes if p['tier']=='accurate' and (p['limit'],p['scenario'],p['initial_context'])==(limit,scenario,initial)]
                for key in CORRECT[:5]:
                    counts={s:sum(p['correctness'][key] for p in items if p['arm']==s) for s in ('main','stack')}
                    target=correctness if scopes.get(scenario,False) else control_diagnostics
                    target.append({'history_scope':'changed-input history' if scopes.get(scenario,False) else 'unchanged-input control' if scenario in scopes else 'input-change scope unproven','configuration':[limit,scenario,initial],'metric':key,**delta(counts['main'],counts['stack'])})
    costs=[]
    for group in sorted({(p['tier'],p['limit'],p['scenario'],p['initial_context']) for p in processes}):
        for key in COST:
            values={s:[p['cost'][key] for p in processes if p['arm']==s and (p['tier'],p['limit'],p['scenario'],p['initial_context'])==group] for s in ('main','stack')}
            means={s:statistics.mean(v) if all(x is not None for x in v) else None for s,v in values.items()}
            costs.append({'configuration':list(group),'metric':key,**delta(means['main'],means['stack'])})
    return {'input_audit_present':audit is not None,'correctness':correctness,'control_diagnostics':control_diagnostics,'metrics':rows,'process_metrics':costs,'disk':[{'configuration':['integration_test_helper_executable' if d['scope']=='app_executable' else d['scope'],d['measure']],'metric':d['measure'],**delta(d['before'],d['after'])} for d in data['disk']] + [{'configuration':['shipped_product_executable','logical_bytes'],'metric':'logical_bytes',**delta(None,None)}]}


def markdown(result):
    def fmt(v):return 'N/A' if v is None else format(v,'.17g') if type(v) in (float,int) else str(v)
    lines=['Sanitized table arithmetic only. Correctness, quality, source/runtime and payload attestations require retained private evidence.',
           ('Input-scope audit is private-attested: 16 paired history cells changed model bytes; 16 were unchanged-input controls. Controls are excluded from changed-input correctness/speed proof.' if result['input_audit_present'] else 'Input-change audit unavailable; changed-input correctness and speed eligibility withheld.'),
           'Four independent process blocks. Accurate hot quantiles pool 20 dependent calls, five per block; Off hot quantiles pool four calls, one per block. Samples per arm are printed for every metric row. CPU zero ticks are resolution-censored.',
           'Native wall has approximately 1Hz exit polling. Cumulative CPU excludes final diagnostic and teardown. RSS/HWM excludes PG; total service N/A.',
           'Same logical index size does not imply no writes. Allowed readonly metadata writes are private-attested.',
           'Legacy app_executable evidence fields measure integration-test helpers including scaffold/source-root strings; shipped product executable disk is N/A.',
           '', '| Configuration | Metric | Before | After | Absolute difference | % change | Samples per arm | Eligibility |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    metrics=sorted(result['metrics'],key=lambda r:(0 if r['metric'].startswith('quality') else 1 if r['metric'] in ('wall_us','cpu_user_seconds','cpu_system_seconds') else 2 if r['metric'] in ('rss_kib','hwm_kib') else 3,str(r['configuration']),r['metric']))
    for r in result['correctness']+result['control_diagnostics']+metrics+result['process_metrics']+result['disk']:
        reason='private-attested/descriptive'
        if 'accuracy_eligible' in r:
            reason='accuracy excluded' if not r['accuracy_eligible'] else 'different recomputation work' if not r['same_successful_work'] else 'stable withheld' if not r['stable_claim_eligible'] else 'four-block descriptive eligible; no statistical proof'
        if r.get('history_scope') in ('unchanged-input control','input-change scope unproven'):
            reason=r['history_scope']+'; not changed-input proof'
        lines.append('| '+' | '.join(['/'.join(map(str,r['configuration'])),r['metric']+' '+r.get('statistic','')]+[fmt(r[k]) for k in ('before','after','absolute_difference','percentage_change')]+[str(r.get('samples_per_arm', 'N/A')),reason])+' |')
    lines += ['', '| Configuration / metric | Four paired block statistics (before, after, delta, %) | Four main medians | Four stack medians | Arm span/median | Sign reversal |', '| --- | --- | --- | --- | --- | --- |']
    for r in metrics:
        blocks=[[b[k] for k in ('before','after','absolute_difference','percentage_change')] for b in r['four_block_statistics']]
        lines.append('| '+' | '.join(['/'.join(map(str,r['configuration']))+'/'+r['metric']+'/'+r['statistic'],json.dumps(blocks,allow_nan=False),json.dumps(r['arm_four_block_medians']['main']),json.dumps(r['arm_four_block_medians']['stack']),json.dumps(r['arm_span_over_median']),str(r['sign_flip'])])+' |')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('evidence');args=parser.parse_args()
    with open(args.evidence,encoding='utf-8') as stream:data=strict_json(stream.read())
    audit_path=Path(args.evidence).with_name('input-scope-audit.json')
    audit=strict_json(audit_path.read_text()) if audit_path.is_file() else None
    if audit is not None:
        require(hashlib.sha256(Path(args.evidence).read_bytes()).hexdigest()==audit['evidence_sha256'], 'input audit exact evidence bytes differ')
    print(markdown(tables(data,audit)),end='')


if __name__=='__main__':main()
