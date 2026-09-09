import importlib.util,sys
from pathlib import Path
import pandas as pd
from cognitive_discovery.mechanistic.dataset.matched_conditions import load_mechanistic_manifest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qwen_confirmation import transformed,calibrate
from llama_interface_validation import InterfaceParticipant

def test_wording_preserves_all_context_statements():
    records=load_mechanistic_manifest('artifacts/qwen_confirmation_v2/conditions.jsonl')
    from cognitive_discovery.causal_mechanistic import pipeline as p
    for r in records:
        original=list(p._trial(r).messages);changed=transformed(original,'rephrased')
        for paragraph in original[-1]['content'].split('Choose one:')[0].strip().split('\n\n'):
            assert paragraph in changed[-1]['content']
        assert original[-1]['content'].split('Choose one:\n')[1].split('\n\n')[0] in changed[-1]['content']

def test_polarity_is_semantically_preserved():
    class Stub:
        model_id='stub';revision='stub'
        def binary_decision(self,messages,labels,*,positive_label):return dict(messages=messages,labels=labels,positive_label=positive_label)
    for variant in ['question','proposition','instruction']:
        for label in ['Yes','No']:
            other='No' if label=='Yes' else 'Yes'
            text=f'State.\n\nChoose one:\n{label} = continue working\n{other} = stop working\n\nRespond with only Yes or No.'
            out=InterfaceParticipant(Stub(),variant).binary_decision([dict(role='user',content=text)],('Yes','No'),positive_label=label)
            assert out['positive_label']==label
            assert 'State.' in out['messages'][0]['content']
            assert ('Would you choose to continue working?' in out['messages'][0]['content'])==(label=='Yes') if variant=='question' else True

def test_original_training_mean_ids_exist():
    pairs=pd.read_parquet('artifacts/causal_mech_v1/counterfactuals/pair_manifest.parquet')
    train=pairs[(pairs.target_variable=='outcome_history')&(pairs.pair_split=='mech_pair_train')]
    ids=set(train.base_condition_id)|set(train.source_condition_id)
    records=load_mechanistic_manifest('artifacts/causal_mech_v1/counterfactuals/mechanistic_conditions.jsonl')
    assert ids and ids.issubset({r.condition.condition_id for r in records})
    assert calibrate()['calibration_rows']==224

def test_cluster_bootstrap_retains_paired_endpoint_difference():
    from analyze_qwen_confirmation import bootstrap,recovery
    frame=pd.DataFrame([dict(task_family=t,background=b,natural_effect=1.,neural_counterfactual_effect=.5,control=0.) for t in ['a','b'] for b in range(4) for _ in range(4)])
    assert recovery(frame.natural_effect,frame.neural_counterfactual_effect)==.75
    assert bootstrap(frame)==[.75,.75]
    assert bootstrap(frame,other='control')==[.75,.75]

def test_mechanistic_llama_question_matches_validated_interface():
    from llama_mechanistic_replication import question_messages
    class Capture:
        model_id='stub';revision='stub'
        def binary_decision(self,messages,labels,*,positive_label):return messages
    for positive in ['Yes','No']:
        negative='No' if positive=='Yes' else 'Yes';messages=[dict(role='user',content=f'History and context.\n\nChoose one:\n{positive} = continue working\n{negative} = stop working\n\nRespond with only Yes or No.')]
        expected=InterfaceParticipant(Capture(),'question').binary_decision(messages,('Yes','No'),positive_label=positive)
        assert question_messages(messages)==expected
