"""Release compatibility for existing low-ratio remainder chains."""
from datetime import date
from itertools import permutations
import pytest
from rtpipeline import organize as org
from test_delivered_plan_selection_authority import _ids, _plan_and_dose, _records

def dose_fixture(tmp,case):
    study,frame,struct=_ids()
    if case=='intended':
        specs=[('anchor',25,50.,3,'20240101',date(2024,1,2),'1.2.826.0.1.3680043.10.999.1'),('transfer',22,44.,22,'20240105',date(2024,1,5),'1.2.826.0.1.3680043.10.999.2'),('boost',8,16.,8,'20240127',date(2024,1,27),'1.2.826.0.1.3680043.10.999.3')]
    else:
        # 25 != 8+16; avoid the separate encompassing-prescription branch.
        # UID order favours 8 fx; date order favours 16 fx. Only one fits after 3 fx.
        same_date=case=='same_date'
        specs=[('anchor',25,50.,3,'20240101',date(2024,1,2),'1.2.826.0.1.3680043.10.999.1'),('short',8,16.,1,'20240106',date(2024,1,5 if same_date else 6),'1.2.826.0.1.3680043.10.999.2'),('long',16,32.,1,'20240105',date(2024,1,5),'1.2.826.0.1.3680043.10.999.3')]
    plans=[];doses=[];records=[]
    for name,fx,rx,delivered,pdate,start,uid in specs:
        plan,dose,_=_plan_and_dose(tmp,name,struct_uid=struct,study_uid=study,frame_uid=frame,plan_date=pdate,rx_gy=rx,fractions=fx,isocentre=(0.,0.,0.),plan_uid=uid)
        plans.append(plan);doses.append(dose);records+=_records(tmp,name+'_rec',uid,delivered,start=start)
    return plans,doses,records


@pytest.mark.parametrize('case', ['different_dates', 'same_date'])
@pytest.mark.parametrize('order', list(permutations(range(3))))
@pytest.mark.parametrize('threshold', [75.0, 100.0])
def test_existing_low_ratio_preserves_release_components_and_warnings(tmp_path, case, order, threshold):
    plans, doses, records = dose_fixture(tmp_path, case)
    result = org._classify_doses([plans[i] for i in order], list(reversed(doses)),
                                treatment_record_paths=records, max_total_dose_gy=threshold)
    assert result.classification == org.DELIVERED_REMAINDER_CHAIN
    assert set(result.selected_plans) == set(plans)
    assert set(result.selected_doses) == set(doses)
    assert [p.stem for p in result.prescription_plans] == ['anchor_plan', 'long_plan']
    assert org._infer_rx_from_plan_paths(result.prescription_plans, sum_all=True) == 82.0
    expected = ['Delivered remainder or adaptation plans retain separate RTDOSE '
                'membership for RTRECORD fraction-weighted course accumulation']
    if threshold < 98.0:
        expected.append(f'PLAUSIBILITY WARNING: selected prescription total 98.0 Gy exceeds {threshold:.1f} Gy')
    assert result.warnings == expected


# Frozen oracle from release 1aeb0c5; independent of the current partition logic.
from typing import List, Dict
_same_fraction_dose = org._same_fraction_dose
def _release_remainder_chain(
    plans: List[dict], delivery: Dict[str, dict]
) -> tuple[dict, List[dict], List[dict], int] | None:
    """Find a likely delivered remainder or adaptation chain.

    Full planned RTDOSE objects from these plans are not directly additive.
    Treatment dates and RTRECORD fraction sessions identify the delivered part
    of each source grid. The classifier retains all contributing plans so the
    course grid can be resampled and weighted by delivered divided by planned
    fractions. The ``< 0.60`` ratio only separates observed early replacement
    chains from independent overlapping phases. Beam-dose equality partitions
    replacement plans from independent phases but never collapses delivered
    plan membership.
    """

    def evidence(plan: dict) -> tuple[set[str], int, int]:
        row = delivery.get(str(plan.get("sop_uid") or ""), {})
        dates = set(row.get("dates", set()))
        delivered = len(row.get("sessions", set()))
        planned = int(plan.get("fractions_planned") or 0)
        return dates, delivered, planned

    supported = [plan for plan in plans if evidence(plan)[1] > 0]
    if len(supported) < 2:
        return None
    planned_total = sum(evidence(plan)[2] for plan in supported)
    delivered_total = sum(evidence(plan)[1] for plan in supported)
    if planned_total <= 0 or delivered_total <= 0 or delivered_total / planned_total >= 0.60:
        return None
    anchor = max(
        supported,
        key=lambda plan: (
            evidence(plan)[2],
            min(evidence(plan)[0]) if evidence(plan)[0] else "",
            str(plan.get("sop_uid") or ""),
        ),
    )
    anchor_dates, anchor_delivered, anchor_planned = evidence(anchor)
    if anchor_planned <= 0 or anchor_delivered >= anchor_planned:
        return None
    later = [plan for plan in supported if plan is not anchor]
    if not later:
        return None
    replacement_plans: List[dict] = []
    independent_phases: List[dict] = []
    represented_fractions = anchor_delivered
    for plan in sorted(later, key=lambda item: str(item.get("sop_uid") or "")):
        planned = evidence(plan)[2]
        if (
            _same_fraction_dose(plan, anchor)
            and represented_fractions + planned <= anchor_planned
        ):
            replacement_plans.append(plan)
            represented_fractions += planned
        else:
            independent_phases.append(plan)
    if replacement_plans:
        return anchor, later, [anchor] + independent_phases, represented_fractions
    if represented_fractions + sum(
        evidence(plan)[2] for plan in independent_phases
    ) <= anchor_planned:
        return anchor, later, [], represented_fractions
    return None

@pytest.mark.parametrize('candidate_fractions', [(8,16), (16,8), (22,8), (30,30), (3,4), (11,11)])
@pytest.mark.parametrize('dates', [('20240106','20240105'), ('20240105','20240105'), ('','')])
@pytest.mark.parametrize('different_dose', [False, True])
def test_low_ratio_partition_matches_release_for_every_input_order(candidate_fractions, dates, different_dose):
    plans = [{'sop_uid':'1', 'fractions_planned':25, 'total_rx_gy':50.0}]
    plans += [{'sop_uid':str(i+2), 'fractions_planned':fx,
               'total_rx_gy':fx*(3.0 if different_dose and i==0 else 2.0)}
              for i,fx in enumerate(candidate_fractions)]
    delivery = {'1':{'sessions':{'a','b','c'},'dates':{'20240102','20240103','20240104'}}}
    for plan, day in zip(plans[1:], dates):
        delivery[plan['sop_uid']] = {'sessions':{'x'}, 'dates':{day} if day else set()}
    for order in permutations(plans):
        assert org._delivered_remainder_chain(list(order), delivery) == _release_remainder_chain(list(order), delivery)
