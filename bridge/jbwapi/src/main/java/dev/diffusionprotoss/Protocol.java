package dev.diffusionprotoss;

import java.util.ArrayList;
import java.util.List;

/**
 * JSON DTOs matching Python's ``starcraft_diffusion_gemma.state.models`` schemas.
 */
public final class Protocol {
    private Protocol() {}

    public static final class TilePointDto {
        public int x;
        public int y;

        public TilePointDto() {}

        public TilePointDto(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public static final class UnitSnapshotDto {
        public int id;
        public String unit_type;
        public int x;
        public int y;
        public int hp;
        public int shields;
        public boolean is_completed;
        public boolean is_under_attack;
        public boolean is_idle;
    }

    public static final class BaseLocationDto {
        public int x;
        public int y;
        public boolean is_start_location;
        public boolean is_occupied_by_us;
        public boolean is_occupied_by_enemy;
    }

    public static final class GameStateDto {
        public int frame;
        public String map_name;
        public int map_width;
        public int map_height;
        public int minerals;
        public int gas;
        public int supply_used;
        public int supply_total;

        public TilePointDto our_base;
        public TilePointDto natural_base;
        public TilePointDto choke_point;
        public TilePointDto enemy_base;

        public List<UnitSnapshotDto> our_units = new ArrayList<>();
        public List<UnitSnapshotDto> our_buildings = new ArrayList<>();
        public List<UnitSnapshotDto> enemy_units = new ArrayList<>();
        public List<UnitSnapshotDto> enemy_buildings = new ArrayList<>();
        public List<BaseLocationDto> expansions = new ArrayList<>();

        public List<String> upgrades_completed = new ArrayList<>();
        public List<String> upgrades_in_progress = new ArrayList<>();
    }

    public static final class BuildStepDto {
        public int supply;
        public String action;
        public String target;
        public int count;
        public String note;
    }

    public static final class CommandBundleDto {
        public int revision = 0;
        public int frame_issued = 0;
        public String posture = "hold_choke";
        public TilePointDto army_target = null;
        public String emergency_action = "none";
        public int target_probes = 22;
        public int target_gateways = 2;
        public double zealot_ratio = 0.4;
        public double dragoon_ratio = 0.6;
        public boolean want_singularity_charge = true;
        public List<String> queued_structures = new ArrayList<>();
        public TilePointDto expand_target = null;
        public BuildStepDto active_build_step = null;
    }
}
