package dev.diffusionprotoss;

import bwapi.*;
import java.util.List;

/**
 * Concurrent S0 Protoss Zealot + Dragoon bot backed by the host S1/S2 policy server.
 *
 * Runs deterministically every BWAPI frame (~24 FPS) without ever waiting on HTTP
 * or vLLM, while syncing GameState snapshots and CommandBundles on a daemon thread.
 */
public final class DiffusionProtossBot extends DefaultBWListener {
    private static final int SYNC_EVERY_FRAMES = 24;
    private static final int BUILD_COOLDOWN_FRAMES = 48;

    private BWClient bwClient;
    private Game game;
    private Player self;
    private PolicySyncClient policyClient;

    private TilePosition enemyBaseGuess = null;
    private int lastBuildOrderFrame = -100;

    public static void main(String[] args) {
        String serverUrl = System.getenv("POLICY_SERVER_URL");
        if (serverUrl == null || serverUrl.isEmpty()) {
            serverUrl = "http://172.18.0.1:8080";
        }
        DiffusionProtossBot bot = new DiffusionProtossBot();
        bot.policyClient = new PolicySyncClient(serverUrl);
        bot.bwClient = new BWClient(bot);
        bot.bwClient.startGame();
    }

    @Override
    public void onStart() {
        game = bwClient.getGame();
        self = game.self();
        game.setLocalSpeed(42);
        // Pick an initial enemy start location guess if 2-player map.
        for (TilePosition start : game.getStartLocations()) {
            if (!start.equals(self.getStartLocation())) {
                enemyBaseGuess = start;
                break;
            }
        }
    }

    @Override
    public void onEnd(boolean isWinner) {
        if (policyClient != null) {
            policyClient.shutdown();
        }
    }

    @Override
    public void onFrame() {
        if (game == null || self == null) {
            return;
        }
        int frame = game.getFrameCount();
        Protocol.CommandBundleDto cmd = policyClient.latestCommand();

        if (frame % SYNC_EVERY_FRAMES == 0) {
            policyClient.submitSnapshot(buildSnapshot(frame));
        }

        runWorkers();
        runMacro(frame, cmd);
        runArmy(cmd);

        game.drawTextScreen(10, 10, "DiffusionProtoss rev=" + cmd.revision
                + " posture=" + cmd.posture + " emergency=" + cmd.emergency_action);
    }

    private void runWorkers() {
        int gasWorkers = 0;
        Unit assimilator = null;
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Assimilator && u.isCompleted()) {
                assimilator = u;
            } else if (u.getType() == UnitType.Protoss_Probe && u.isGatheringGas()) {
                gasWorkers++;
            }
        }
        for (Unit u : self.getUnits()) {
            if (u.getType() != UnitType.Protoss_Probe || !u.isCompleted()) {
                continue;
            }
            if (assimilator != null && gasWorkers < 3 && u.isGatheringMinerals()) {
                u.gather(assimilator);
                gasWorkers++;
                continue;
            }
            if (u.isIdle()) {
                Unit closestMineral = null;
                int bestDist = Integer.MAX_VALUE;
                for (Unit neutral : game.neutral().getUnits()) {
                    if (neutral.getType().isMineralField()) {
                        int d = u.getDistance(neutral);
                        if (d < bestDist) {
                            bestDist = d;
                            closestMineral = neutral;
                        }
                    }
                }
                if (closestMineral != null) {
                    u.gather(closestMineral);
                }
            }
        }
    }

    private void runMacro(int frame, Protocol.CommandBundleDto cmd) {
        int probes = countUnits(UnitType.Protoss_Probe);
        int gateways = countUnits(UnitType.Protoss_Gateway);
        int zealots = countUnits(UnitType.Protoss_Zealot);
        int dragoons = countUnits(UnitType.Protoss_Dragoon);
        int supplyUsed = self.supplyUsed() / 2;
        int supplyTotal = self.supplyTotal() / 2;

        // 1. Continuous Probe production up to target_probes.
        if (probes < cmd.target_probes && self.minerals() >= 50) {
            for (Unit u : self.getUnits()) {
                if (u.getType() == UnitType.Protoss_Nexus && u.isCompleted() && !u.isTraining()) {
                    u.train(UnitType.Protoss_Probe);
                    break;
                }
            }
        }

        // 2. Emergency or proactive Pylon to avoid supply block.
        boolean needPylon = "emergency_pylon".equals(cmd.emergency_action)
                || (supplyTotal < 200 && supplyTotal - supplyUsed <= 3 && !isConstructing(UnitType.Protoss_Pylon));
        if (needPylon && self.minerals() >= 100 && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
            if (tryBuild(UnitType.Protoss_Pylon, self.getStartLocation())) {
                lastBuildOrderFrame = frame;
                return;
            }
        }

        // 3. Follow active opening BuildStep when supply threshold is reached.
        if (cmd.active_build_step != null && supplyUsed >= cmd.active_build_step.supply) {
            if ("build".equals(cmd.active_build_step.action)
                    && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
                UnitType toBuild = mapStructure(cmd.active_build_step.target);
                if (toBuild != null && self.minerals() >= toBuild.mineralPrice()
                        && self.gas() >= toBuild.gasPrice()
                        && !isConstructing(toBuild)) {
                    if (tryBuild(toBuild, self.getStartLocation())) {
                        lastBuildOrderFrame = frame;
                        return;
                    }
                }
            }
        }

        // 4. Scale Gateways up to target_gateways after opening.
        if (cmd.active_build_step == null && gateways < cmd.target_gateways
                && self.minerals() >= 150 && !isConstructing(UnitType.Protoss_Gateway)
                && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
            if (tryBuild(UnitType.Protoss_Gateway, self.getStartLocation())) {
                lastBuildOrderFrame = frame;
                return;
            }
        }

        // 5. Research Singularity Charge at Cybernetics Core.
        if (cmd.want_singularity_charge
                && self.getUpgradeLevel(UpgradeType.Singularity_Charge) == 0
                && !self.isUpgrading(UpgradeType.Singularity_Charge)
                && self.minerals() >= 150 && self.gas() >= 150) {
            for (Unit u : self.getUnits()) {
                if (u.getType() == UnitType.Protoss_Cybernetics_Core
                        && u.isCompleted() && !u.isUpgrading()) {
                    u.upgrade(UpgradeType.Singularity_Charge);
                    break;
                }
            }
        }

        // 6. Train Zealots / Dragoons from idle Gateways per S2 ratio.
        boolean hasCore = countCompleted(UnitType.Protoss_Cybernetics_Core) > 0;
        for (Unit u : self.getUnits()) {
            if (u.getType() != UnitType.Protoss_Gateway || !u.isCompleted() || u.isTraining()) {
                continue;
            }
            boolean preferDragoon = hasCore
                    && !"emergency_zealot".equals(cmd.emergency_action)
                    && self.minerals() >= 125 && self.gas() >= 50
                    && (zealots + dragoons == 0
                        || ((double) dragoons / Math.max(1, zealots + dragoons)) < cmd.dragoon_ratio);
            if (preferDragoon) {
                u.train(UnitType.Protoss_Dragoon);
            } else if (self.minerals() >= 100) {
                u.train(UnitType.Protoss_Zealot);
            }
        }
    }

    private void runArmy(Protocol.CommandBundleDto cmd) {
        Position target = null;
        if (cmd.army_target != null) {
            target = new TilePosition(cmd.army_target.x, cmd.army_target.y).toPosition();
        } else if ("push".equals(cmd.posture) && enemyBaseGuess != null) {
            target = enemyBaseGuess.toPosition();
        } else {
            target = self.getStartLocation().toPosition();
        }

        for (Unit u : self.getUnits()) {
            if (!u.isCompleted()) {
                continue;
            }
            UnitType t = u.getType();
            boolean isCombat = (t == UnitType.Protoss_Zealot || t == UnitType.Protoss_Dragoon);
            if (!isCombat) {
                continue;
            }
            if ("retreat".equals(cmd.posture)) {
                Position home = self.getStartLocation().toPosition();
                if (u.getDistance(home) > 256 && !u.isMoving()) {
                    u.move(home);
                }
            } else if (u.isIdle() && u.getDistance(target) > 192) {
                u.attack(target);
            }
        }
    }

    private boolean tryBuild(UnitType type, TilePosition near) {
        Unit builder = null;
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Probe && u.isCompleted()
                    && (u.isGatheringMinerals() || u.isIdle())) {
                builder = u;
                break;
            }
        }
        if (builder == null) {
            return false;
        }
        TilePosition loc = game.getBuildLocation(type, near, 32, false);
        if (loc == null || !loc.isValid(game)) {
            return false;
        }
        return builder.build(type, loc);
    }

    private int countUnits(UnitType type) {
        int c = 0;
        for (Unit u : self.getUnits()) {
            if (u.getType() == type) {
                c++;
            }
        }
        return c;
    }

    private int countCompleted(UnitType type) {
        int c = 0;
        for (Unit u : self.getUnits()) {
            if (u.getType() == type && u.isCompleted()) {
                c++;
            }
        }
        return c;
    }

    private boolean isConstructing(UnitType type) {
        for (Unit u : self.getUnits()) {
            if (u.getType() == type && !u.isCompleted()) {
                return true;
            }
            if (u.getType() == UnitType.Protoss_Probe && u.getBuildType() == type) {
                return true;
            }
        }
        return false;
    }

    private static UnitType mapStructure(String name) {
        if (name == null) {
            return null;
        }
        switch (name.toLowerCase()) {
            case "pylon":
                return UnitType.Protoss_Pylon;
            case "gateway":
                return UnitType.Protoss_Gateway;
            case "assimilator":
                return UnitType.Protoss_Assimilator;
            case "cybernetics_core":
                return UnitType.Protoss_Cybernetics_Core;
            case "nexus":
                return UnitType.Protoss_Nexus;
            default:
                return null;
        }
    }

    private static String normalizeType(UnitType t) {
        String raw = t.toString().toLowerCase();
        if (raw.startsWith("protoss_")) {
            return raw.substring("protoss_".length());
        }
        if (raw.startsWith("terran_")) {
            return raw.substring("terran_".length());
        }
        if (raw.startsWith("zerg_")) {
            return raw.substring("zerg_".length());
        }
        return raw;
    }

    private Protocol.GameStateDto buildSnapshot(int frame) {
        Protocol.GameStateDto dto = new Protocol.GameStateDto();
        dto.frame = frame;
        dto.map_name = game.mapFileName();
        dto.map_width = game.mapWidth();
        dto.map_height = game.mapHeight();
        dto.minerals = self.minerals();
        dto.gas = self.gas();
        dto.supply_used = self.supplyUsed() / 2;
        dto.supply_total = self.supplyTotal() / 2;

        TilePosition start = self.getStartLocation();
        dto.our_base = new Protocol.TilePointDto(start.getX(), start.getY());
        if (enemyBaseGuess != null) {
            dto.enemy_base = new Protocol.TilePointDto(enemyBaseGuess.getX(), enemyBaseGuess.getY());
        }

        for (Unit u : self.getUnits()) {
            Protocol.UnitSnapshotDto snap = toUnitDto(u);
            if (u.getType().isBuilding()) {
                dto.our_buildings.add(snap);
            } else {
                dto.our_units.add(snap);
            }
        }
        List<Unit> enemies = game.enemy().getUnits();
        for (Unit e : enemies) {
            if (!e.isVisible()) {
                continue;
            }
            Protocol.UnitSnapshotDto snap = toUnitDto(e);
            if (e.getType().isBuilding()) {
                dto.enemy_buildings.add(snap);
                if (e.getType().isResourceDepot()) {
                    TilePosition tp = e.getTilePosition();
                    enemyBaseGuess = tp;
                    dto.enemy_base = new Protocol.TilePointDto(tp.getX(), tp.getY());
                }
            } else {
                dto.enemy_units.add(snap);
            }
        }
        if (self.getUpgradeLevel(UpgradeType.Singularity_Charge) > 0) {
            dto.upgrades_completed.add("singularity_charge");
        } else if (self.isUpgrading(UpgradeType.Singularity_Charge)) {
            dto.upgrades_in_progress.add("singularity_charge");
        }
        return dto;
    }

    private static Protocol.UnitSnapshotDto toUnitDto(Unit u) {
        Protocol.UnitSnapshotDto snap = new Protocol.UnitSnapshotDto();
        snap.id = u.getID();
        snap.unit_type = normalizeType(u.getType());
        TilePosition tp = u.getTilePosition();
        snap.x = tp.getX();
        snap.y = tp.getY();
        snap.hp = Math.max(0, u.getHitPoints());
        snap.shields = Math.max(0, u.getShields());
        snap.is_completed = u.isCompleted();
        snap.is_under_attack = u.isUnderAttack();
        snap.is_idle = u.isIdle();
        return snap;
    }
}
