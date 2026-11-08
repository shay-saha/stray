#pragma once

#include "simulation.hpp"

#include <cstdint>
#include <filesystem>
#include <imgui.h>
#include <memory>

namespace wildlife {

struct EarthTileView {
    float x = 0.0f;
    float y = 0.0f;
    float side = 0.0f;
};

class EarthTiles {
public:
    EarthTiles(const std::filesystem::path& executable, bool stream);
    ~EarthTiles();

    EarthTileView update(ImVec2 center, float radius, ImVec2 min, ImVec2 max,
                         Vec3 east, Vec3 north, Vec3 forward);
    float elevation(EarthTileView view, double u, double sin_latitude, float fallback, bool land_only = true) const;
    Vec3 color(EarthTileView view, double u, double sin_latitude, Vec3 fallback) const;
    std::uint64_t revision() const;
    void bind(unsigned int fallback_texture) const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace wildlife
