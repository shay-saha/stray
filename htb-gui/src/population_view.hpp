#pragma once

#include "globe.hpp"

#include <optional>

namespace wildlife {

struct AnimalColorFamily {
    const char *label;
    ImU32 color;
};

std::span<const AnimalColorFamily> animal_color_key();
ImU32 species_color(const Species &species);
ImU32 animal_color(const Species &species, std::string_view identity);
ImFontAtlasRectId make_grain_sprite(ImFontAtlas &atlas);
std::optional<AnimalId> draw_populations(ImDrawList &draw, const GlobeView &view, const Simulation &simulation,
                                         std::span<const std::uint8_t> enabled, float pixels_per_km,
                                         double history_hours, std::optional<AnimalId> selected,
                                         std::optional<ImVec2> pointer,
                                         ImFontAtlasRectId grain_sprite = ImFontAtlasRectId_Invalid);

} // namespace wildlife
