package dev.diffusionprotoss;

import bwapi.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Concurrent S0 Protoss Zealot + Dragoon bot backed by the host S1/S2 policy server.
 *
 * Runs deterministically every BWAPI frame (~24 FPS) without ever waiting on HTTP
 * or vLLM, while syncing GameState snapshots and CommandBundles on a daemon thread.
 */
public final class DiffusionProtossBot extends DefaultBWListener {
    private static final int SYNC_EVERY_FRAMES = 24;
    private static final int BUILD_COOLDOWN_FRAMES = 48;
    private static final int GAS_BALANCE_EVERY_FRAMES = 12;
    private static final int ARMY_SYNC_EVERY_FRAMES = 24;
    private static final int LEASH_RADIUS_PX = 18 * 32;
    private static final int WORKER_CHASE_LEASH_PX = 10 * 32;

    private BWClient bwClient;
    private Game game;
    private Player self;
    private PolicySyncClient policyClient;

    private final List<TilePosition> baseLocations = new ArrayList<>();
    private final List<TilePosition> unscoutedStarts = new ArrayList<>();
    private TilePosition naturalBase = null;
    private TilePosition chokePoint = null;
    private TilePosition enemyBaseGuess = null;

    private int lastBuildOrderFrame = -100;
    private String lastArmyPosture = "";
    private TilePosition lastArmyTargetTile = null;

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

        discoverBaseLocations();

        TilePosition myStart = self.getStartLocation();
        for (TilePosition start : game.getStartLocations()) {
            if (!start.equals(myStart)) {
                unscoutedStarts.add(start);
            }
        }
        if (!unscoutedStarts.isEmpty()) {
            enemyBaseGuess = unscoutedStarts.get(0);
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

        runWorkers(frame);
        runMacro(frame, cmd);
        runArmy(frame, cmd);

        game.drawTextScreen(
                10,
                10,
                "DiffusionProtoss rev=" + cmd.revision
                        + " posture=" + cmd.posture
                        + " leash=" + cmd.leash_army
                        + " expand=" + (cmd.expand_target != null
                                ? "(" + cmd.expand_target.x + "," + cmd.expand_target.y + ")"
                                : "none")
                        + " emergency=" + cmd.emergency_action);
    }

    /**
     * Cluster static mineral fields on map start to identify start locations,
     * expansion Nexus tiles, the natural expansion, and an approximate rally/choke tile.
     */
    private void discoverBaseLocations() {
        baseLocations.clear();
        TilePosition myStart = self.getStartLocation();
        for (TilePosition start : game.getStartLocations()) {
            baseLocations.add(start);
        }

        List<List<Unit>> clusters = new ArrayList<>();
        for (Unit mineral : game.getStaticMinerals()) {
            if (mineral.getInitialResources() <= 0) {
                continue;
            }
            Position mPos = mineral.getInitialPosition();
            List<Unit> matched = null;
            for (List<Unit> cluster : clusters) {
                Position cPos = cluster.get(0).getInitialPosition();
                if (mPos.getDistance(cPos) <= 9 * 32) {
                    matched = cluster;
                    break;
                }
            }
            if (matched == null) {
                matched = new ArrayList<>();
                clusters.add(matched);
            }
            matched.add(mineral);
        }

        for (List<Unit> cluster : clusters) {
            if (cluster.size() < 4) {
                continue;
            }
            int sumX = 0;
            int sumY = 0;
            for (Unit m : cluster) {
                TilePosition tp = m.getInitialTilePosition();
                sumX += tp.getX();
                sumY += tp.getY();
            }
            int cx = Math.round((float) sumX / cluster.size());
            int cy = Math.round((float) sumY / cluster.size());
            TilePosition centroid = new TilePosition(cx, cy);

            boolean alreadyKnown = false;
            for (TilePosition existing : baseLocations) {
                if (tileDist(existing, centroid) <= 10.0) {
                    alreadyKnown = true;
                    break;
                }
            }
            if (alreadyKnown) {
                continue;
            }

            TilePosition snapped = findBestNexusTileNear(centroid);
            if (snapped != null) {
                baseLocations.add(snapped);
            }
        }

        double bestNatDist = Double.MAX_VALUE;
        for (TilePosition base : baseLocations) {
            double d = tileDist(base, myStart);
            if (d > 10.0 && d < bestNatDist) {
                bestNatDist = d;
                naturalBase = base;
            }
        }

        TilePosition mapCenter = new TilePosition(game.mapWidth() / 2, game.mapHeight() / 2);
        if (naturalBase != null) {
            int cx = (int) Math.round(naturalBase.getX() * 0.75 + mapCenter.getX() * 0.25);
            int cy = (int) Math.round(naturalBase.getY() * 0.75 + mapCenter.getY() * 0.25);
            chokePoint = clampTile(cx, cy);
        } else {
            int cx = (int) Math.round(myStart.getX() * 0.75 + mapCenter.getX() * 0.25);
            int cy = (int) Math.round(myStart.getY() * 0.75 + mapCenter.getY() * 0.25);
            chokePoint = clampTile(cx, cy);
        }
    }

    private TilePosition findBestNexusTileNear(TilePosition center) {
        TilePosition best = null;
        double bestDist = Double.MAX_VALUE;
        for (int dx = -8; dx <= 8; dx++) {
            for (int dy = -8; dy <= 8; dy++) {
                TilePosition cand = new TilePosition(center.getX() + dx, center.getY() + dy);
                if (!cand.isValid(game)) {
                    continue;
                }
                if (!game.canBuildHere(cand, UnitType.Protoss_Nexus)) {
                    continue;
                }
                double d = tileDist(cand, center);
                if (d < bestDist) {
                    bestDist = d;
                    best = cand;
                }
            }
        }
        return best;
    }

    private void runWorkers(int frame) {
        // 1. Every frame: assign any idle probe to the best mineral patch.
        for (Unit u : self.getUnits()) {
            if (u.getType() != UnitType.Protoss_Probe || !u.isCompleted()) {
                continue;
            }
            if (u.isIdle()) {
                sendToBestMineral(u);
            }
        }

        // 2. Periodically enforce strict 3-probe saturation per completed Assimilator.
        //    Running every 12 frames (0.5s) avoids BWAPI command-latency double-assignment.
        if (frame % GAS_BALANCE_EVERY_FRAMES != 0) {
            return;
        }

        List<Unit> assimilators = new ArrayList<>();
        List<Unit> probes = new ArrayList<>();
        for (Unit u : self.getUnits()) {
            if (!u.isCompleted()) {
                continue;
            }
            if (u.getType() == UnitType.Protoss_Assimilator) {
                assimilators.add(u);
            } else if (u.getType() == UnitType.Protoss_Probe) {
                probes.add(u);
            }
        }
        if (assimilators.isEmpty()) {
            return;
        }

        Map<Integer, List<Unit>> gasByAssim = new HashMap<>();
        for (Unit a : assimilators) {
            gasByAssim.put(a.getID(), new ArrayList<>());
        }

        List<Unit> availableMineralProbes = new ArrayList<>();
        for (Unit p : probes) {
            if (isAssignedToGas(p)) {
                Unit targetAssim = resolveProbeAssimilator(p, assimilators);
                if (targetAssim != null) {
                    gasByAssim.get(targetAssim.getID()).add(p);
                }
            } else if (p.isGatheringMinerals() && !p.isConstructing() && !p.isCarryingMinerals()) {
                availableMineralProbes.add(p);
            }
        }

        Set<Integer> reassignedIds = new HashSet<>();
        for (Unit assim : assimilators) {
            List<Unit> assigned = gasByAssim.get(assim.getID());
            if (assigned.size() > 3) {
                // Pull excess probes off gas, preferring ones not carrying gas.
                assigned.sort(Comparator.comparing(Unit::isCarryingGas));
                int excess = assigned.size() - 3;
                for (int i = 0; i < excess; i++) {
                    sendToBestMineral(assigned.get(i));
                }
            } else if (assigned.size() < 3) {
                int needed = 3 - assigned.size();
                availableMineralProbes.sort(Comparator.comparingInt(p -> p.getDistance(assim)));
                for (Unit p : availableMineralProbes) {
                    if (needed <= 0) {
                        break;
                    }
                    if (reassignedIds.contains(p.getID())) {
                        continue;
                    }
                    p.gather(assim);
                    reassignedIds.add(p.getID());
                    needed--;
                }
            }
        }
    }

    private static boolean isAssignedToGas(Unit probe) {
        if (probe.isGatheringGas()) {
            return true;
        }
        Order o = probe.getOrder();
        return o == Order.MoveToGas
                || o == Order.WaitForGas
                || o == Order.HarvestGas
                || o == Order.ReturnGas;
    }

    private static Unit resolveProbeAssimilator(Unit probe, List<Unit> assimilators) {
        Unit target = probe.getOrderTarget();
        if (target != null && target.getType() == UnitType.Protoss_Assimilator) {
            for (Unit a : assimilators) {
                if (a.getID() == target.getID()) {
                    return a;
                }
            }
        }
        Unit nearest = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit a : assimilators) {
            int d = probe.getDistance(a);
            if (d < bestDist) {
                bestDist = d;
                nearest = a;
            }
        }
        return nearest;
    }

    private void sendToBestMineral(Unit probe) {
        List<Unit> completedNexuses = new ArrayList<>();
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Nexus && u.isCompleted()) {
                completedNexuses.add(u);
            }
        }

        // Prefer a Nexus whose mineral line is under-saturated (< 16 probes nearby).
        Unit targetNexus = null;
        int bestScore = Integer.MAX_VALUE;
        for (Unit nexus : completedNexuses) {
            int nearbyWorkers = 0;
            for (Unit u : self.getUnits()) {
                if (u.getType() == UnitType.Protoss_Probe
                        && u.isGatheringMinerals()
                        && u.getDistance(nexus) <= 12 * 32) {
                    nearbyWorkers++;
                }
            }
            if (nearbyWorkers < 16) {
                int score = nearbyWorkers * 64 + probe.getDistance(nexus) / 4;
                if (score < bestScore) {
                    bestScore = score;
                    targetNexus = nexus;
                }
            }
        }

        Position anchor = targetNexus != null ? targetNexus.getPosition() : probe.getPosition();
        Unit closestMineral = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit neutral : game.neutral().getUnits()) {
            if (neutral.getType().isMineralField()) {
                int d = anchor.getApproxDistance(neutral.getPosition());
                if (d < bestDist) {
                    bestDist = d;
                    closestMineral = neutral;
                }
            }
        }
        if (closestMineral != null) {
            probe.gather(closestMineral);
        }
    }

    private void runMacro(int frame, Protocol.CommandBundleDto cmd) {
        int probes = countUnits(UnitType.Protoss_Probe);
        int gateways = countUnits(UnitType.Protoss_Gateway);
        int completedGateways = countCompleted(UnitType.Protoss_Gateway);
        int completedNexuses = countCompleted(UnitType.Protoss_Nexus);
        int zealots = countUnits(UnitType.Protoss_Zealot);
        int dragoons = countUnits(UnitType.Protoss_Dragoon);
        int supplyUsed = self.supplyUsed() / 2;
        int supplyTotal = self.supplyTotal() / 2;

        // 1. Continuous Probe production across all idle completed Nexuses up to target_probes.
        if (probes < cmd.target_probes && self.minerals() >= 50) {
            for (Unit u : self.getUnits()) {
                if (u.getType() == UnitType.Protoss_Nexus && u.isCompleted() && !u.isTraining()) {
                    if (self.minerals() >= 50) {
                        u.train(UnitType.Protoss_Probe);
                        probes++;
                        if (probes >= cmd.target_probes) {
                            break;
                        }
                    }
                }
            }
        }

        // 2. Emergency or proactive Pylon with dynamic headroom based on active production.
        int supplyHeadroom = Math.min(10, Math.max(3, 2 + 2 * completedGateways + completedNexuses));
        int pylonsInProgress = countConstructing(UnitType.Protoss_Pylon);
        int maxConcurrentPylons = (completedGateways >= 4 && supplyTotal - supplyUsed <= 4) ? 2 : 1;
        boolean needPylon = "emergency_pylon".equals(cmd.emergency_action)
                || (supplyTotal < 200
                        && supplyTotal - supplyUsed <= supplyHeadroom
                        && pylonsInProgress < maxConcurrentPylons);
        if (needPylon && self.minerals() >= 100 && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
            if (tryBuild(UnitType.Protoss_Pylon, self.getStartLocation())) {
                lastBuildOrderFrame = frame;
                return;
            }
        }

        // 3. S2-driven expansion: reserve minerals and construct Nexus at snapped expand_target.
        boolean reservingForExpand = false;
        if (cmd.expand_target != null) {
            TilePosition expandTile = snapToExpansionTile(
                    new TilePosition(cmd.expand_target.x, cmd.expand_target.y));
            if (expandTile != null && !hasNexusNear(expandTile, 10.0) && !isConstructing(UnitType.Protoss_Nexus)) {
                reservingForExpand = true;
                if (self.minerals() >= 400 && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
                    if (tryBuildNexusAt(expandTile)) {
                        lastBuildOrderFrame = frame;
                        return;
                    }
                }
            }
        }

        // 4. Follow active opening BuildStep when supply threshold is reached.
        if (!reservingForExpand
                && cmd.active_build_step != null
                && supplyUsed >= cmd.active_build_step.supply) {
            if ("build".equals(cmd.active_build_step.action)
                    && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
                UnitType toBuild = mapStructure(cmd.active_build_step.target);
                if (toBuild != null && self.minerals() >= toBuild.mineralPrice()
                        && self.gas() >= toBuild.gasPrice()
                        && !isConstructing(toBuild)) {
                    if (tryBuildStructure(toBuild, self.getStartLocation())) {
                        lastBuildOrderFrame = frame;
                        return;
                    }
                }
            }
        }

        // 5. Execute S2 queued_structures (one-off structures beyond the opening).
        if (!reservingForExpand
                && cmd.queued_structures != null
                && !cmd.queued_structures.isEmpty()
                && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
            UnitType queued = mapStructure(cmd.queued_structures.get(0));
            if (queued != null
                    && self.minerals() >= queued.mineralPrice()
                    && self.gas() >= queued.gasPrice()
                    && !isConstructing(queued)) {
                boolean built;
                if (queued == UnitType.Protoss_Nexus) {
                    TilePosition exp = pickNextFreeExpansion();
                    built = exp != null && tryBuildNexusAt(exp);
                } else {
                    built = tryBuildStructure(queued, self.getStartLocation());
                }
                if (built) {
                    lastBuildOrderFrame = frame;
                    return;
                }
            }
        }

        // 6. Scale Gateways up to target_gateways after opening (unless reserving for expansion).
        if (!reservingForExpand
                && cmd.active_build_step == null
                && gateways < cmd.target_gateways
                && self.minerals() >= 150
                && !isConstructing(UnitType.Protoss_Gateway)
                && frame - lastBuildOrderFrame > BUILD_COOLDOWN_FRAMES) {
            if (tryBuild(UnitType.Protoss_Gateway, self.getStartLocation())) {
                lastBuildOrderFrame = frame;
                return;
            }
        }

        // 7. Research Singularity Charge at Cybernetics Core.
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

        // 8. Train Zealots / Dragoons from idle Gateways per S2 ratio.
        //    While reserving for a Nexus, only spend surplus minerals above 400 unless emergency.
        boolean allowUnitSpend = !reservingForExpand
                || "emergency_zealot".equals(cmd.emergency_action)
                || self.minerals() >= 500;
        if (!allowUnitSpend) {
            return;
        }
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

    private void runArmy(int frame, Protocol.CommandBundleDto cmd) {
        String posture = cmd.posture != null ? cmd.posture : "hold_choke";
        boolean isPush = "push".equals(posture);
        boolean isRetreat = "retreat".equals(posture);

        TilePosition targetTile;
        if (cmd.army_target != null) {
            targetTile = new TilePosition(cmd.army_target.x, cmd.army_target.y);
            // If pushing and army_target is still near our main, push toward enemyBaseGuess.
            if (isPush && enemyBaseGuess != null
                    && tileDist(targetTile, self.getStartLocation()) < 16.0) {
                targetTile = enemyBaseGuess;
            }
        } else if (isPush && enemyBaseGuess != null) {
            targetTile = enemyBaseGuess;
        } else if (chokePoint != null) {
            targetTile = chokePoint;
        } else {
            targetTile = self.getStartLocation();
        }
        Position target = targetTile.toPosition();

        // On 4-player maps, cycle enemyBaseGuess if our pushing army reached an empty start.
        if (isPush && enemyBaseGuess != null && unscoutedStarts.size() > 1) {
            Position guessPos = enemyBaseGuess.toPosition();
            if (hasFriendlyCombatNear(guessPos, 192) && !hasVisibleEnemyNear(guessPos, 320)) {
                unscoutedStarts.remove(enemyBaseGuess);
                if (!unscoutedStarts.isEmpty()) {
                    enemyBaseGuess = unscoutedStarts.get(0);
                    targetTile = enemyBaseGuess;
                    target = targetTile.toPosition();
                }
            }
        }

        boolean postureChanged = !posture.equals(lastArmyPosture);
        boolean targetChanged = lastArmyTargetTile == null
                || tileDist(targetTile, lastArmyTargetTile) >= 8.0;
        boolean periodicSync = (frame % ARMY_SYNC_EVERY_FRAMES == 0);
        if (postureChanged || targetChanged) {
            lastArmyPosture = posture;
            lastArmyTargetTile = targetTile;
        }

        Position home = self.getStartLocation().toPosition();
        for (Unit u : self.getUnits()) {
            if (!u.isCompleted()) {
                continue;
            }
            UnitType t = u.getType();
            if (t != UnitType.Protoss_Zealot && t != UnitType.Protoss_Dragoon) {
                continue;
            }

            if (isRetreat) {
                if (u.getDistance(home) > 256 && (!u.isMoving() || postureChanged)) {
                    u.move(home);
                }
                continue;
            }

            if (isPush) {
                // Actively commit non-idle units when switching to push or when far from target.
                if (postureChanged || targetChanged) {
                    if (!u.isAttackFrame()) {
                        u.attack(target);
                    }
                } else if (u.isIdle() && u.getDistance(target) > 192) {
                    u.attack(target);
                } else if (periodicSync && !u.isAttacking() && u.getDistance(target) > 256) {
                    u.attack(target);
                }
                continue;
            }

            // Defensive / rally posture: enforce S2-controlled leash to prevent SCV chasing.
            if (cmd.leash_army && periodicSync && shouldLeashUnit(u, target)) {
                u.move(target);
                continue;
            }

            if ((postureChanged || targetChanged) && !u.isAttackFrame() && u.getDistance(target) > 256) {
                u.attack(target);
            } else if (u.isIdle() && u.getDistance(target) > 192) {
                u.attack(target);
            }
        }
    }

    private boolean shouldLeashUnit(Unit u, Position target) {
        int distToTarget = u.getDistance(target);
        // Never yank a unit out of a genuine fight against an enemy combat unit near our bases.
        if (hasEnemyCombatNear(u.getPosition(), 8 * 32) && distToTarget <= LEASH_RADIUS_PX) {
            return false;
        }
        // Pull back any unit that drifted far beyond the defensive leash radius.
        if (distToTarget > LEASH_RADIUS_PX && !hasNexusWithinPx(u.getPosition(), 14 * 32)) {
            return true;
        }
        // Pull back any unit chasing a lone enemy worker away from our rally/choke.
        Unit orderTarget = u.getOrderTarget();
        if (orderTarget != null
                && orderTarget.exists()
                && orderTarget.getType().isWorker()
                && distToTarget > WORKER_CHASE_LEASH_PX
                && !hasEnemyCombatNear(u.getPosition(), 10 * 32)) {
            return true;
        }
        return false;
    }

    private boolean hasEnemyCombatNear(Position pos, int radiusPx) {
        for (Unit e : game.enemy().getUnits()) {
            if (e.isVisible() && !e.getType().isWorker() && !e.getType().isBuilding()
                    && e.getDistance(pos) <= radiusPx) {
                return true;
            }
        }
        return false;
    }

    private boolean hasVisibleEnemyNear(Position pos, int radiusPx) {
        for (Unit e : game.enemy().getUnits()) {
            if (e.isVisible() && e.getDistance(pos) <= radiusPx) {
                return true;
            }
        }
        return false;
    }

    private boolean hasFriendlyCombatNear(Position pos, int radiusPx) {
        for (Unit u : self.getUnits()) {
            UnitType t = u.getType();
            if (u.isCompleted()
                    && (t == UnitType.Protoss_Zealot || t == UnitType.Protoss_Dragoon)
                    && u.getDistance(pos) <= radiusPx) {
                return true;
            }
        }
        return false;
    }

    private boolean hasNexusWithinPx(Position pos, int radiusPx) {
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Nexus && u.getDistance(pos) <= radiusPx) {
                return true;
            }
        }
        return false;
    }

    private boolean hasNexusNear(TilePosition tile, double maxTiles) {
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Nexus && tileDist(u.getTilePosition(), tile) <= maxTiles) {
                return true;
            }
        }
        return false;
    }

    private TilePosition snapToExpansionTile(TilePosition requested) {
        TilePosition best = null;
        double bestDist = Double.MAX_VALUE;
        for (TilePosition base : baseLocations) {
            if (hasNexusNear(base, 8.0)) {
                continue;
            }
            double d = tileDist(base, requested);
            if (d < bestDist) {
                bestDist = d;
                best = base;
            }
        }
        if (best != null && bestDist <= 20.0) {
            return best;
        }
        return findBestNexusTileNear(requested);
    }

    private TilePosition pickNextFreeExpansion() {
        TilePosition myStart = self.getStartLocation();
        TilePosition best = null;
        double bestDist = Double.MAX_VALUE;
        for (TilePosition base : baseLocations) {
            if (hasNexusNear(base, 8.0)) {
                continue;
            }
            double d = tileDist(base, myStart);
            if (d < bestDist) {
                bestDist = d;
                best = base;
            }
        }
        return best;
    }

    private boolean tryBuildNexusAt(TilePosition tile) {
        Unit builder = pickBuilderNear(tile.toPosition());
        if (builder == null) {
            return false;
        }
        if (game.canBuildHere(tile, UnitType.Protoss_Nexus)) {
            return builder.build(UnitType.Protoss_Nexus, tile);
        }
        TilePosition fallback = game.getBuildLocation(UnitType.Protoss_Nexus, tile, 16, false);
        if (fallback != null && fallback.isValid(game)) {
            return builder.build(UnitType.Protoss_Nexus, fallback);
        }
        return false;
    }

    private boolean tryBuildStructure(UnitType type, TilePosition near) {
        if (type == UnitType.Protoss_Assimilator) {
            return tryBuildAssimilator();
        }
        return tryBuild(type, near);
    }

    private boolean tryBuildAssimilator() {
        for (Unit nexus : self.getUnits()) {
            if (nexus.getType() != UnitType.Protoss_Nexus || !nexus.isCompleted()) {
                continue;
            }
            for (Unit geyser : game.getGeysers()) {
                if (geyser.getDistance(nexus) <= 12 * 32) {
                    TilePosition loc = geyser.getTilePosition();
                    Unit builder = pickBuilderNear(geyser.getPosition());
                    if (builder != null && builder.build(UnitType.Protoss_Assimilator, loc)) {
                        return true;
                    }
                }
            }
        }
        return tryBuild(UnitType.Protoss_Assimilator, self.getStartLocation());
    }

    private boolean tryBuild(UnitType type, TilePosition near) {
        Unit builder = pickBuilderNear(near.toPosition());
        if (builder == null) {
            return false;
        }
        TilePosition loc = game.getBuildLocation(type, near, 32, false);
        if (loc == null || !loc.isValid(game)) {
            return false;
        }
        return builder.build(type, loc);
    }

    private Unit pickBuilderNear(Position pos) {
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Probe && u.isCompleted()
                    && (u.isGatheringMinerals() || u.isIdle())
                    && !u.isConstructing()
                    && !u.isCarryingMinerals()) {
                int d = u.getDistance(pos);
                if (d < bestDist) {
                    bestDist = d;
                    best = u;
                }
            }
        }
        if (best != null) {
            return best;
        }
        for (Unit u : self.getUnits()) {
            if (u.getType() == UnitType.Protoss_Probe && u.isCompleted()
                    && (u.isGatheringMinerals() || u.isIdle())
                    && !u.isConstructing()) {
                return u;
            }
        }
        return null;
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

    private int countConstructing(UnitType type) {
        int c = 0;
        for (Unit u : self.getUnits()) {
            if (u.getType() == type && !u.isCompleted()) {
                c++;
            } else if (u.getType() == UnitType.Protoss_Probe && u.getBuildType() == type) {
                c++;
            }
        }
        return c;
    }

    private boolean isConstructing(UnitType type) {
        return countConstructing(type) > 0;
    }

    private TilePosition clampTile(int x, int y) {
        int cx = Math.max(0, Math.min(game.mapWidth() - 1, x));
        int cy = Math.max(0, Math.min(game.mapHeight() - 1, y));
        return new TilePosition(cx, cy);
    }

    private static double tileDist(TilePosition a, TilePosition b) {
        int dx = a.getX() - b.getX();
        int dy = a.getY() - b.getY();
        return Math.sqrt(dx * dx + dy * dy);
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
        if (naturalBase != null) {
            dto.natural_base = new Protocol.TilePointDto(naturalBase.getX(), naturalBase.getY());
        }
        if (chokePoint != null) {
            dto.choke_point = new Protocol.TilePointDto(chokePoint.getX(), chokePoint.getY());
        }
        if (enemyBaseGuess != null) {
            dto.enemy_base = new Protocol.TilePointDto(enemyBaseGuess.getX(), enemyBaseGuess.getY());
        }

        int mineralWorkers = 0;
        int gasWorkers = 0;
        int idleWorkers = 0;
        for (Unit u : self.getUnits()) {
            Protocol.UnitSnapshotDto snap = toUnitDto(u);
            if (u.getType().isBuilding()) {
                dto.our_buildings.add(snap);
            } else {
                dto.our_units.add(snap);
                if (u.getType() == UnitType.Protoss_Probe && u.isCompleted()) {
                    if (isAssignedToGas(u)) {
                        gasWorkers++;
                    } else if (u.isGatheringMinerals()) {
                        mineralWorkers++;
                    } else if (u.isIdle()) {
                        idleWorkers++;
                    }
                }
            }
        }
        dto.mineral_workers = mineralWorkers;
        dto.gas_workers = gasWorkers;
        dto.idle_workers = idleWorkers;

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

        for (TilePosition base : baseLocations) {
            Protocol.BaseLocationDto b = new Protocol.BaseLocationDto();
            b.x = base.getX();
            b.y = base.getY();
            b.is_start_location = game.getStartLocations().contains(base);
            b.is_occupied_by_us = hasNexusNear(base, 8.0);
            boolean enemyHere = false;
            for (Unit e : enemies) {
                if (e.isVisible() && e.getType().isResourceDepot()
                        && tileDist(e.getTilePosition(), base) <= 8.0) {
                    enemyHere = true;
                    break;
                }
            }
            if (!enemyHere && enemyBaseGuess != null && tileDist(enemyBaseGuess, base) <= 6.0) {
                enemyHere = true;
            }
            b.is_occupied_by_enemy = enemyHere;
            dto.expansions.add(b);
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
