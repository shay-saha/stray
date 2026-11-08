"""Metadata-only, coverage-first public GPS intake plan; never downloads GPS.

All catalogue records are audited. The finite queue uses taxon, animal-count,
source-reference, title-region and temporal hints; no forecast outcomes enter.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('migration_acquire', PROJECT / 'scripts/migration_acquire.py')
acquire = importlib.util.module_from_spec(spec); spec.loader.exec_module(acquire)

DOMESTIC = {'Bos taurus', 'Canis familiaris', 'Capra hircus', 'Felis catus', 'Ovis aries'}
NON_LOCATION = re.compile(r'video|movie|acceleration|accelerometer|magnetometer|orientation|gyroscope|barometer|(?:^|[-_ .])(?:acc\d*(?:of\d+)?|tdr)(?:[-_ .]|$)|heart.?rate', re.I)
REGIONS = {'Europe':r'\b(?:europe|uk|united kingdom|britain|scotland|germany|france|netherlands|spain|portugal|italy|sweden|norway|finland|poland|kosov[ao]|switzerland|alps|denmark|iceland|svalbard|mediterranean)\b',
 'Asia':r'\b(?:asia|china|mongolia|bhutan|india|thailand|japan|laos|cambodia|nepal|indonesia|malaysia|singapore|russia|siberia|bangladesh|israel|taiwan)\b',
 'Africa':r'\b(?:africa|kenya|namibia|tanzania|botswana|zimbabwe|zambia|uganda|congo|ethiopia|senegal|mauritania|mozambique|malawi|laikipia|pilanesberg)\b',
 'North America':r'\b(?:usa|u\.s\.|united states|canada|mexico|alaska|california|wyoming|colorado|florida|banff|alberta|quebec|montana|ontario|nova scotia|north america)\b',
 'South America':r'\b(?:south america|brazil|brasil|colombia|ecuador|peru|bolivia|argentina|chile|venezuela|gal[aá]pagos|uruguay|paraguay)\b',
 'Central America/Caribbean':r'\b(?:panama|costa rica|belize|honduras|nicaragua|guatemala|caribbean|cuba|jamaica|haiti|puerto rico|bocas del toro)\b',
 'Oceania':r'\b(?:australia|new zealand|tasmania|papua|queensland|victoria|fiji)\b',
 'Antarctic/Southern Ocean':r'\b(?:antarctic|antarctica|southern ocean|crozet|kerguelen)\b',
 'Arctic':r'\b(?:arctic|greenland)\b',
 'Ocean':r'\b(?:atlantic|pacific|indian ocean|north sea|baltic|chagos)\b'}


def sha(path):
    return acquire.digest(path)


def checksum(f):
    c = f.get('published_checksum') or {}
    return (str(c.get('checkSumAlgorithm','')).casefold(), str(c.get('value','')).casefold())


def doi(value):
    match = re.search(r'10\.\d{4,9}/[^\s<>"#]+', unquote(value), re.I)
    return match.group(0).rstrip('.,/').casefold() if match else None


def inventory(metadata):
    """Recover complete ORIGINAL metadata, preserving safe basename collisions."""
    source = json.loads((metadata / 'bitstreams.json').read_text())
    grouped = {}
    for response in source['http_responses']:
        path = Path(response['path'])
        if sha(path) != response['decoded_response_sha256']:
            raise ValueError('API response checksum mismatch')
        body = json.loads(path.read_text())
        for value in body.get('_embedded',{}).get('bitstreams',[]):
            name = acquire.safe_name(value['name'])
            grouped.setdefault(response['source_id'],[]).append({'name':name,'original_name':value['name'],
                'bytes':int(value['sizeBytes']),'url':value['_links']['content']['href'],
                'published_checksum':value.get('checkSum'),'bitstream_uuid':value['uuid'],
                'inventory_response_sha256':response['decoded_response_sha256']})
    result = {}
    for source_id, files in grouped.items():
        seen, names, selected = set(), {}, []
        for f in sorted(files,key=lambda f:f['bitstream_uuid']):
            key = (f['name'],f['bytes'],checksum(f))
            if key in seen:
                continue
            seen.add(key); names.setdefault(f['name'],[]).append(f)
        for name, same in sorted(names.items()):
            for f in same:
                if len(same)>1:
                    p=Path(name); f['name']=p.stem+'__'+f['bitstream_uuid'][:8]+p.suffix
                    f['basename_collision_resolution']='Distinct ORIGINAL versions preserved by UUID suffix; original_name retained'
                selected.append(f)
        result[source_id] = selected
    return result, source


def role(f):
    name=f['original_name'].casefold()
    if name.startswith(('readme','license','licence')):
        return 'source_documentation'
    if NON_LOCATION.search(name) or acquire.NON_GPS.search(name):
        return 'non_location_sensor_or_media'
    return acquire.file_role(f)


def temporal(title, files):
    """Source-time hints only; never confuse publication time with tracking time."""
    text=title+' '+' '.join(f['original_name'] for f in files if role(f) in {'events','archive'})
    ranges=[(int(a),int(b)) for a,b in re.findall(r'\b((?:19|20)\d{2})\s*(?:[-–—/]|to|through)\s*((?:19|20)\d{2})\b',text)]
    ranges=[(a,b) for a,b in ranges if 1900<=a<=b<=2026]
    years=sorted({int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b',text) if 1900<=int(y)<=2026})
    post_only=bool(years) and min(years)>=2022
    pre_hint=bool(years) and min(years)<=2020
    status='explicit_title_file_years_only_2022plus' if post_only else 'pre2021_title_file_hint' if pre_hint else '2021_only_hint' if years else 'tracking_years_unknown'
    span='multiyear_metadata_hint' if any(b>a for a,b in ranges) else 'seasonal_topic_unverified' if re.search(r'migrat|annual|year.round|season',title,re.I) else 'short_or_local_topic_unverified' if re.search(r'forag|home.range|hour|daily|short.term|day|commut',title,re.I) else 'span_unknown'
    return {'status':status,'years_in_title_or_event_names':years,'year_ranges':ranges,
            'span_hint':span,'counts_as_confirmed_pre2021_training':False,
            'excluded_from_potential_2020_training_taxa':post_only or status=='2021_only_hint',
            'warning':'Tracking dates remain unverified. Publication year is not tracking year; actual GPS clocks and spans require intake QA.'}


def scope_status(taxa, title):
    if 'Homo sapiens' in taxa:
        return 'human_scope_hold'
    if (taxa and set(taxa)<=DOMESTIC) or re.search(r'\b(?:domestic ducks|homing pigeons?|pet cats|pet dogs)\b',title,re.I):
        return 'domestic_scope_hold'
    return 'wildlife_candidate_unverified'


def build(catalogue, metadata, output, budget=30_000_000_000, max_file=2_000_000_000):
    catalogue,metadata,output=Path(catalogue),Path(metadata),Path(output)
    if output.exists():raise FileExistsError('The breadth plan is immutable; choose a new path')
    cat=json.loads(catalogue.read_text()); files, raw_inventory=inventory(metadata)
    if sha(catalogue)!=raw_inventory['catalogue_sha256']:raise ValueError('Catalogue/inventory fingerprint mismatch')
    cache=json.loads((metadata/'cache-inventory.json').read_text()); cached={}
    for r in cache['records']:
        key=checksum(r)
        if key[0] and key[1]:cached.setdefault(key,[]).append(r)
    for key in cached:cached[key].sort(key=lambda r:(r['cache_location']!='training_server',r['path']))
    manifest=json.loads((PROJECT/'data/migration-expanded-dev-v3/manifest.json').read_text())
    old_train=set(manifest.get('species',{})); old_reporting=set(manifest.get('metric_species',{}))
    primary=json.loads((metadata/'movebench-studies.json').read_text()); primary_taxa={r['species'] for r in primary['rows']}
    primary_dois={doi(u) for r in primary['rows'] for s in r['source_refs'] for u in s['urls'] if doi(u)}
    audit=[]; candidates=[]
    for rec in cat['records']:
        m=rec['metadata']; taxa=sorted(set(t.strip() for t in m.get('dwc.ScientificName',[]) if t.strip()))
        gps=any(v.strip().casefold()=='gps' for v in m.get('sensor.name',[]))
        rights=acquire.rights_status(m); eligible=gps and rights=='eligible'
        fs=files.get(rec['id'],[]); possible=[f for f in fs if role(f) in {'events','archive'} and f['bytes']<=max_file]
        # Direct CSVs avoid uncertain ZIP expansion wherever they cover a source.
        events=[f for f in possible if role(f)=='events'] or [f for f in possible if role(f)=='archive']
        supporting=[f for f in fs if (role(f)=='identity_reference' and f['bytes']<=max_file)
                    or (role(f)=='source_documentation' and f['bytes']<=min(max_file,2_000_000))]
        scope=scope_status(taxa,rec['title'])
        regions=sorted(k for k,pattern in REGIONS.items() if re.search(pattern,rec['title'],re.I))
        time=temporal(rec['title'],fs)
        record={'id':rec['id'],'title':rec['title'],'url':rec['url'],'scientific_names_metadata':taxa,
            'rights_status':rights,'rights_uris':m.get('dc.rights.uri',[]),'sensors_metadata':m.get('sensor.name',[]),
            'gps_explicit':gps,'eligible_gps_metadata':eligible,'scope':scope,
            'scope_flags':['domestic_taxon:'+t for t in taxa if t in DOMESTIC]+(['wild_or_feral_horse_requires_scope_audit'] if 'Equus ferus' in taxa else []),
            'animals_metadata_not_independent':acquire.number(m,'mdr.animal.count'),
            'locations_metadata_not_ranked':acquire.number(m,'mdr.location.count'),
            'study_ids_metadata':m.get('mdr.study.id',[]),'regions_title_hint':regions,'temporal_metadata':time,
            'movebench_species_string_overlap':sorted(set(taxa)&primary_taxa),
            'movebench_data_doi_match':bool({doi(v) for v in m.get('dc.identifier.doi',[])} & primary_dois),
            'processing_status':'QA hold until actual GPS, modelled/interpolated/smoothed flags, source processing, deployments and nominal-clock provenance are audited',
            'files':[]}
        for f in fs:
            f={**f,'role':role(f)};matches=cached.get(checksum(f),[])
            matches=[r for r in matches if r['bytes']==f['bytes']]
            f['cached_matches']=matches
            f['included_for_intake']=any(f['bitstream_uuid']==v['bitstream_uuid'] for v in events+supporting)
            if f['included_for_intake']:
                # These are per-file hard output reservations, not claims about
                # unknown ZIP expansion. Intake must reject files that exceed it.
                extracted=2*f['bytes']+65536 if f['role']=='events' else min(2_000_000_000,max(32_000_000,20*f['bytes'])) if f['role']=='archive' else 0
                original=0 if matches and matches[0]['cache_location']=='training_server' else f['bytes']
                f.update(extracted_hard_cap_bytes=extracted,original_storage_reservation_bytes=original,
                    storage_reservation_bytes=original+extracted,
                    network_download_bytes=0 if matches else f['bytes'],
                    action='reuse_verified_server_original' if matches and matches[0]['cache_location']=='training_server' else 'transfer_reverify_local_original' if matches else 'public_checksum_verified_download_after_approval')
            record['files'].append(f)
        record['storage_reservation_bytes']=sum(f.get('storage_reservation_bytes',0) for f in record['files'])
        record['network_download_bytes']=sum(f.get('network_download_bytes',0) for f in record['files'])
        record['status']='eligible_pending_QA' if eligible and events and scope=='wildlife_candidate_unverified' else 'scope_hold' if eligible and scope!='wildlife_candidate_unverified' else 'no_supported_location_candidate' if eligible else 'licence_hold' if rights!='eligible' else 'no_explicit_GPS_metadata'
        audit.append(record)
        if record['status']=='eligible_pending_QA':candidates.append(record)
    selected=[];covered=set(old_train);regions=set();spans={};used=0
    while True:
        available=[r for r in candidates if r['id'] not in {x['id'] for x in selected}
                   and set(r['scientific_names_metadata'])-covered and used+r['storage_reservation_bytes']<=budget]
        if not available:break
        def priority(r):
            t=r['temporal_metadata']; temporal_rank=2 if t['status']=='pre2021_title_file_hint' else 0 if t['excluded_from_potential_2020_training_taxa'] else 1
            return (-len(set(r['scientific_names_metadata'])-covered),-temporal_rank,
                -min(r['animals_metadata_not_independent'],20),
                -len(set(r['regions_title_hint'])-regions),
                -int(any(f['role']=='identity_reference' and f['included_for_intake'] for f in r['files'])),
                spans.get(t['span_hint'],0),r['storage_reservation_bytes'],r['id'])
        chosen=min(available,key=priority)
        chosen['selection_rank']=len(selected)+1
        chosen['new_taxa_at_selection']=sorted(set(chosen['scientific_names_metadata'])-covered)
        selected.append(chosen); covered.update(chosen['scientific_names_metadata']); regions.update(chosen['regions_title_hint'])
        spans[chosen['temporal_metadata']['span_hint']]=spans.get(chosen['temporal_metadata']['span_hint'],0)+1
        used+=chosen['storage_reservation_bytes']
    wildlife={t for r in audit if r['eligible_gps_metadata'] and r['scope']=='wildlife_candidate_unverified' for t in r['scientific_names_metadata']}
    potential_train=set(old_train)
    for r in selected:
        if not r['temporal_metadata']['excluded_from_potential_2020_training_taxa']:potential_train.update(r['scientific_names_metadata'])
    result={'format':'wildcast-breadth-metadata-plan-v1','scope':'Metadata-only finite acquisition plan; no GPS files downloaded, no training, no official benchmark or independent-animal count claim',
        'catalogue_sha256':sha(catalogue),'inventory_sha256':sha(metadata/'bitstreams.json'),
        'cache_inventory_sha256':sha(metadata/'cache-inventory.json'),'movebench_html_sha256':sha(metadata/'movebench.html'),
        'movebench_studies_sha256':sha(metadata/'movebench-studies.json'),'code_sha256':sha(__file__),
        'catalogue_packages_audited':len(audit),'gps_cc0_by_packages':sum(r['eligible_gps_metadata'] for r in audit),
        'gps_cc0_by_scientific_name_strings':sorted({t for r in audit if r['eligible_gps_metadata'] for t in r['scientific_names_metadata']}),
        'wildlife_candidate_scientific_name_strings':sorted(wildlife),
        'frozen_v3_training_taxa':sorted(old_train),'frozen_v3_reporting_taxa':sorted(old_reporting),
        'selected_sources':selected,'selected_count':len(selected),'selected_new_taxa':sorted(covered-old_train),
        'available_training_plus_selected_taxa_metadata':sorted(covered),
        'potential_2020_training_taxa_excluding_explicit_new_only':sorted(potential_train),
        'confirmed_new_training_taxa':[],
        'quotas':{'original_plus_extracted_hard_cap_bytes':budget,'maximum_original_file_bytes':max_file,
            'minimum_server_free_bytes':50_000_000_000,'server_free_bytes_at_inventory':cache['training_server_free_bytes'],
            'planned_storage_reservation_bytes':used,'planned_external_download_bytes':sum(r['network_download_bytes'] for r in selected),
            'remaining_reservation_bytes':budget-used,'metadata_fetch_workers':4},
        'span_hint_source_counts':spans,'region_title_hints':sorted(regions),
        'movebench':{'source':'https://arxiv.org/html/2609.15780v1#A2.T3','abstract_species_claim':110,
            'table_study_rows':len(primary['rows']),'table_distinct_latin_strings':len(primary_taxa),
            'table_study_individual_sum_not_deduplicated':sum(r['curated_individuals'] for r in primary['rows']),
            'official_train_test_manifests_available':False,'contamination_rule':'Any eventual official score requires clean retraining from official training manifests. Raw-source overlap and locally trained weights cannot establish official unseen performance.'},
        'execution_contract':{'approved_to_download':False,'raw_root':'data/migration-breadth-stage1/raw',
            'layout':'<source_uuid>/source.json, originals/<safe filename>, lossless GPS candidate CSV chunks and identity references beside source.json',
            'immutable_originals':True,'verify_published_checksum_and_SHA256':True,
            'reuse_cached_originals_only_after_rehash':True,'enforce_per_file_extracted_caps_and_shared_total_budget':True,
            'check_free_floor_before_each_file_and_chunk':True,'generic_migration_cleaner_allowed':False,
            'intake_QA':['actual GPS sensor rows only','reject explicitly modelled/interpolated/smoothed GPS','hold unclear source processing','preserve raw taxonomy and animal/tag/deployment IDs','preserve nominal clocks; never assert UTC precision without source evidence','frozen-v3 physical holdout assignment survives new aliases','audit actual pre2021 GPS availability before counting training taxa','do not count raw fixes as long-horizon target labels']},
        'all_packages':audit}
    if used>budget or cache['training_server_free_bytes']-used<50_000_000_000:raise ValueError('Plan exceeds storage/free-floor contract')
    acquire.save(output,result)
    print(json.dumps({'plan':str(output),'audited':len(audit),'eligible_gps_packages':result['gps_cc0_by_packages'],
        'eligible_gps_latin_strings':len(result['gps_cc0_by_scientific_name_strings']),
        'wildlife_candidate_latin_strings':len(wildlife),'selected_sources':len(selected),
        'training_plus_selected_metadata_taxa':len(covered),'potential_pre2021_training_upper_bound':len(potential_train),
        'new_selected_taxa':len(covered-old_train),'storage_reservation_GB':used/1e9,
        'external_download_GB':result['quotas']['planned_external_download_bytes']/1e9},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue',default='data/catalogue.json');p.add_argument('--metadata',default='data/migration-breadth-stage1/metadata')
    p.add_argument('--output',default='data/migration-breadth-stage1/plan.json')
    build(**vars(p.parse_args()))

if __name__=='__main__':main()
