#pragma once

#include "simulation.hpp"

#include <filesystem>
#include <imgui.h>

namespace wildlife {

inline constexpr double earth_radius_meters = 6371000.0;
inline constexpr double max_land_elevation = 9000.0;

bool initialize_earth_texture(const std::filesystem::path& executable, bool stream_tiles = true);
void shutdown_earth_texture();
bool earth_terrain_available();
bool earth_detail_visible();
float earth_surface_height(Vec3 point, bool detail = true);

struct LocalSurface {
    float height; // Land or water surface, in meters above sea level.
    float water_depth;
    Vec3 color;
};

// Update the same tile cache used by the globe. Coordinates for samples are in radians.
std::uint64_t prepare_local_terrain(Vec3 anchor, float half_extent_m);
LocalSurface sample_local_terrain(double latitude, double longitude, bool detail = true);
bool draw_earth_texture(ImDrawList& draw, ImVec2 center, float radius, Vec3 east, Vec3 north, Vec3 forward,
                        Vec3 light, float pivot_y, float pivot_radius);

} // namespace wildlife
