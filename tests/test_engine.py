# -*- coding: utf-8 -*-
"""末日地堡生存 —— 核心引擎的可测试纯逻辑，验证资源守恒、危机决策、结局判定。

注意：测试使用独立内存级 Session，需清空表。为隔离，这里用 engine 建临时表。
"""
import pytest
from sqlalchemy.orm import Session

from app.core.database import Base, engine, SessionLocal
from app.core.config import INITIAL_RESOURCES, SURVIVAL_TARGET_DAY
from app.models import GameSession, Resident, Facility
from app.services.engine import (
    BunkerEngine,
    BunkerEngineError,
    CRISIS_POOL,
    FACILITY_ZH,
    FOOD,
    OXY,
    POWER,
    WATER,
)


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    s = SessionLocal()
    yield s
    s.close()
    Base.metadata.drop_all(bind=engine)


def make_session(db, residents=3, resources=None):
    gs = GameSession(
        name="测试",
        day=1,
        target_day=SURVIVAL_TARGET_DAY,
        status="running",
        resources=resources or dict(INITIAL_RESOURCES),
        survivors=residents,
        score=0,
    )
    db.add(gs)
    db.flush()
    for i in range(residents):
        db.add(Resident(session_id=gs.id, name=f"人{i}", job="general", health=90, morale=80, alive=1, joined_day=1))
    for cat in ("power", "farm", "water", "oxygen"):
        db.add(Facility(session_id=gs.id, name=FACILITY_ZH[cat], category=cat, level=1, status="active", built_day=1))
    db.commit()
    db.refresh(gs)
    return gs


class FixedRand:
    """固定值随机 —— 每个 .random() 返回 0.9（不触发危机，因 0.9 > 0.45）。"""

    def random(self):
        return 0.9

    def choice(self, seq):
        return seq[0]


def test_advance_increments_day(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.advance_day()
    assert gs.day == 2


def test_resources_change_with_population(db):
    """资源应有产出-消耗的净变化（守恒循环运行）。"""
    gs = make_session(db, residents=3)
    before = dict(gs.resources)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.advance_day()
    after = gs.resources
    # 至少一个资源发生变化
    assert any(abs(after[k] - before[k]) > 0.01 for k in ("food", "water", "power", "oxygen"))


def test_build_deducts_cost(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    food_before = gs.resources[FOOD]
    eng.build_facility("med")
    assert gs.resources[FOOD] < food_before
    assert any(f.category == "med" for f in gs.facilities)


def test_build_fails_when_poor(db):
    gs = make_session(db)
    gs.resources = {FOOD: 1, WATER: 1, POWER: 1, OXY: 1}
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.build_facility("farm")


def test_upgrade_increases_level(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    fac = [f for f in gs.facilities if f.category == "farm"][0]
    eng.upgrade_facility(fac.id)
    assert fac.level == 2


def test_crisis_applies_resource_effects(db):
    """选择翻倍食物选项应减食物。"""
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    # 直接调用危机池里第一个事件的效果
    event = CRISIS_POOL[0]
    food_before = gs.resources[FOOD]
    eng._apply_crisis(event)  # 仅生成待决策
    choice = event["choices"][0]
    eff = choice["effects"].get("resources", {}).get(FOOD, 0)
    eng.resolve_crisis(event["key"], choice["key"])
    assert gs.resources[FOOD] <= food_before + eff + 1


def test_job_assignment_changes_resident(db):
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    r = gs.residents[0]
    eng.set_job(r.id, "farmer")
    assert r.job == "farmer"


def test_win_at_target_day(db):
    gs = make_session(db, resources={FOOD: 9999, WATER: 9999, POWER: 9999, OXY: 9999})
    gs.day = SURVIVAL_TARGET_DAY  # 目标天数
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng._check_end()
    assert gs.status == "win"


def test_population_zero_ends_game(db):
    gs = make_session(db)
    for r in gs.residents:
        r.alive = 0
    gs.survivors = 0
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng._check_end()
    assert gs.status == "over"


def test_advance_rejected_after_game_end(db):
    gs = make_session(db)
    gs.status = "over"
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.advance_day()


def test_morale_recovery_toward_75(db):
    gs = make_session(db)
    for r in gs.residents:
        r.morale = 40
    gs.resources = {FOOD: 999, WATER: 999, POWER: 999, OXY: 999}
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng._apply_health_morale()
    assert all(r.morale > 40 for r in gs.residents)


class AlwaysCrisis:
    """必触发危机的随机源（0.0 <= 0.45），choice 取第一个。"""

    def random(self):
        return 0.0

    def choice(self, seq):
        return seq[0]


# ---- 危机结算目标校验（跨档案污染回归）----

def test_resolve_crisis_rejects_foreign_resident(db):
    """提交其他档案的居民编号：拒绝结算，当前档案数据不受污染。"""
    gs1 = make_session(db)
    gs2 = make_session(db)
    foreign = gs2.residents[0]
    eng = BunkerEngine(db, gs1, rand=FixedRand())
    res_before = dict(gs1.resources)
    state_before = [(r.health, r.morale) for r in gs1.residents]
    with pytest.raises(BunkerEngineError):
        eng.resolve_crisis("sick", "quarantine", target_id=foreign.id)
    assert gs1.resources == res_before
    assert [(r.health, r.morale) for r in gs1.residents] == state_before


def test_resolve_crisis_rejects_unknown_resident(db):
    """不存在的居民编号同样按无效目标拒绝。"""
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.resolve_crisis("sick", "quarantine", target_id=999999)


def test_resolve_crisis_rejects_dead_resident(db):
    """已故居民是无效目标，不应再被结算。"""
    gs = make_session(db)
    dead = gs.residents[0]
    dead.alive = 0
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.resolve_crisis("sick", "quarantine", target_id=dead.id)


def test_resolve_crisis_hits_only_target(db):
    """合法目标：健康/士气效果只落在该居民身上。"""
    gs = make_session(db)
    target = gs.residents[1]
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.resolve_crisis("sick", "quarantine", target_id=target.id)  # health -5
    assert target.health == 85
    others = [r for r in gs.residents if r.id != target.id]
    assert all(r.health == 90 for r in others)


def test_resolve_crisis_without_target_hits_all_alive(db):
    """未指定目标时保持既有语义：效果作用于全体存活居民。"""
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    eng.resolve_crisis("mutiny", "double_ration")  # morale +20
    assert all(r.morale == 100 for r in gs.residents)  # 80+20，封顶 100


def test_resolve_crisis_rejected_after_game_end(db):
    """结算边界：档案已结局后不允许再结算危机。"""
    gs = make_session(db)
    gs.status = "win"
    eng = BunkerEngine(db, gs, rand=FixedRand())
    with pytest.raises(BunkerEngineError):
        eng.resolve_crisis("sick", "quarantine")


def test_no_crisis_issued_on_settlement_day(db):
    """到达结局的当天不再派发新危机（结算后无可决议事件）。"""
    gs = make_session(db, resources={FOOD: 9999, WATER: 9999, POWER: 9999, OXY: 9999})
    gs.day = SURVIVAL_TARGET_DAY - 1
    eng = BunkerEngine(db, gs, rand=AlwaysCrisis())
    crisis = eng.advance_day()
    assert gs.status == "win"
    assert crisis is None


def test_crisis_still_issued_mid_game(db):
    """未结局时危机照常触发，且目标来自当前档案。"""
    gs = make_session(db)
    eng = BunkerEngine(db, gs, rand=AlwaysCrisis())
    crisis = eng.advance_day()
    assert crisis is not None
    assert crisis["target_id"] in [r.id for r in gs.residents]