#pragma once

#include "globe.hpp"

#include <filesystem>

namespace wildlife {

struct MapLabelRect {
    ImVec2 min;
    ImVec2 max;
};

ImFont* load_habitat_font(const std::filesystem::path& executable);
void draw_habitat_labels(ImDrawList& draw, const GlobeView& globe, ImFont* font, MapLabelRect bounds,
                         float zoom, std::span<const MapLabelRect> blocked);

} // namespace wildlife
